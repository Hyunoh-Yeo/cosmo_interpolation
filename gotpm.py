"""Reader for GOTPM / Multiverse DM snapshot files -- the ONE place the format lives.

Every analysis script imports from here; none re-implement the decode. On the
cluster, scp this file alongside the script you run (two files, still standalone).

Format confirmed against real data (2026-07, MV_Om0.26_-1_0/INITIAL.0118000000):
  * ASCII header of variable length, terminated by the line '#End of Ascii Header'
  * then packed particle records, 32 bytes each, LITTLE-ENDIAN (no byte swap):
        float32 x, y, z, vx, vy, vz   +   int64 indx
  * x,y,z are DISPLACEMENTS from the Lagrangian grid node (grid units); the absolute
    comoving position is
        pos = fmod(displacement + grid_node_from(indx), n) * (boxsize / n)
    where indx is the GLOBAL Lagrangian index (0 .. Nx*Ny*Nz-1), shared across all
    Multiverse cosmologies (same initial conditions) -- this is what lets us line
    particles up between cosmologies for interpolation.

A snapshot is split into Nid (=250) subfiles named  <prefix>.<step:05d><sub:05d>,
e.g. INITIAL.0118000000 .. INITIAL.0118000249  (step 1180).

Velocities are read as raw code units; the km/s conversion factor (Pfact / Fact*)
is still to be pinned down from the GOTPM source -- positions are exact.
"""
import os
import glob
import numpy as np

_CLOSING = "#End of Ascii Header"
RECORD = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4"),
                   ("indx", "<i8")])
assert RECORD.itemsize == 32


def read_header(path):
    """Return (params dict, byte offset where binary particle data begins)."""
    params = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: EOF before header end")
            text = line.decode("latin-1", "replace").rstrip("\n")
            if text.strip() == _CLOSING:
                return params, f.tell()
            if text.startswith("define") and "=" in text:
                key, val = text[len("define"):].split("=", 1)
                params[key.strip()] = val.strip()


def geom(params):
    """(nx, ny, nz, box[cMpc/h], mx, mxmy) as numbers, pulled from the header."""
    nx = int(params["Nx"]); ny = int(params["Ny"]); nz = int(params["Nz"])
    box = float(params["Boxsize(Mpc/h)"])
    mx = int(params["Mx"]); mxmy = int(params["MxMy"])
    return nx, ny, nz, box, mx, mxmy


def redshift(params):
    """z of a snapshot from its header (Amax = 48)."""
    return 48.0 / float(params["Anow"]) - 1.0


def read_records(path, stride=1, id_mod=None, frac=None, rng=None):
    """Return (structured record array, params).

    stride>1  memory-maps and keeps every Nth record, so the OS only faults in the
              pages it touches (a 4 KiB page holds 128 records, so stride>=128 also
              cuts I/O; e.g. stride=1280 -> ~10x less read).
    id_mod    keep only records with indx % id_mod == 0 (an identity-based sample:
              picks the SAME Lagrangian particles in every cosmology).
    frac      keep a uniform random fraction (needs rng); still reads the whole file.
    """
    params, off = read_header(path)
    if stride > 1:
        n = (os.path.getsize(path) - off) // RECORD.itemsize
        rec = np.array(np.memmap(path, dtype=RECORD, mode="r", offset=off, shape=(n,))[::stride])
    else:
        with open(path, "rb") as f:
            f.seek(off)
            rec = np.fromfile(f, dtype=RECORD)
    if id_mod:
        rec = rec[rec["indx"] % id_mod == 0]
    if frac is not None and frac < 1.0:
        rec = rec[rng.random(rec.size) < frac]
    return rec, params


def decode_positions(rec, params):
    """Records -> absolute comoving positions [N,3] in cMpc/h.

    indx encodes the Lagrangian grid node (indx = iz*MxMy + iy*Mx + ix); the stored
    x/y/z are the displacement from that node. pos = (node + displacement) wrapped
    into the box, scaled to cMpc/h. This decode is the physical core of the project.
    """
    nx, ny, nz, box, mx, mxmy = geom(params)
    indx = rec["indx"]
    ix = indx % mx
    iy = (indx % mxmy) // mx
    iz = indx // mxmy
    pos = np.empty((rec.size, 3), np.float32)
    pos[:, 0] = np.mod(rec["x"] + ix, nx) * (box / nx)
    pos[:, 1] = np.mod(rec["y"] + iy, ny) * (box / ny)
    pos[:, 2] = np.mod(rec["z"] + iz, nz) * (box / nz)
    return pos


def read_subfile(path, want_pos=True, want_vel=False, stride=1, id_mod=None,
                 frac=None, rng=None):
    """Read one subfile -> dict(indx, pos[N,3] cMpc/h, vel[N,3] raw, header, n).

    `pos`/`vel` are only computed if requested (saves memory on huge files).
    stride/id_mod/frac are forwarded to read_records (subsampling / identity slice).
    """
    rec, params = read_records(path, stride=stride, id_mod=id_mod, frac=frac, rng=rng)
    out = {"indx": rec["indx"], "header": params, "n": rec.size}
    if want_pos:
        out["pos"] = decode_positions(rec, params)
    if want_vel:
        out["vel"] = np.stack([rec["vx"], rec["vy"], rec["vz"]], axis=1)  # raw units
    return out


def list_subfiles(cosmo_dir, step, prefix="INITIAL"):
    """Sorted list of subfile paths for a snapshot step, e.g. step=1180."""
    pat = os.path.join(cosmo_dir, f"{prefix}.{step:05d}?????")
    return sorted(glob.glob(pat))


def snapshot_steps(cosmo_dir, prefix="INITIAL"):
    """Unique snapshot step numbers present in a cosmology directory."""
    steps = set()
    for p in glob.glob(os.path.join(cosmo_dir, f"{prefix}.??????????")):
        tag = os.path.basename(p).split(".")[-1]      # 10 digits: step(5)+sub(5)
        if len(tag) == 10 and tag.isdigit():
            steps.add(int(tag[:5]))
    return sorted(steps)


def iter_snapshot(cosmo_dir, step, prefix="INITIAL", want_pos=True, want_vel=False,
                  max_sub=None, **kw):
    """Yield per-subfile dicts for a whole snapshot (stream, don't hold all at once).

    Use this to process the ~275 GB/snapshot in tiles. `max_sub` limits how many
    subfiles are read (handy for a quick test). Extra kwargs go to read_subfile.
    """
    files = list_subfiles(cosmo_dir, step, prefix)
    if max_sub is not None:
        files = files[:max_sub]
    for p in files:
        yield read_subfile(p, want_pos=want_pos, want_vel=want_vel, **kw)


def read_snapshot(cosmo_dir, step, prefix="INITIAL", want_pos=True, want_vel=False,
                  max_sub=None, sort_by_index=False, **kw):
    """Read (and concatenate) a snapshot. WARNING: a full snapshot is ~2048^3
    particles (~275 GB) -- pass max_sub for tests, or use iter_snapshot for tiling.

    With sort_by_index=True the particles are returned ordered by global index, so
    row i is the same Lagrangian particle in every cosmology (enables interpolation).
    """
    idx, pos, vel = [], [], []
    for d in iter_snapshot(cosmo_dir, step, prefix, want_pos, want_vel, max_sub, **kw):
        idx.append(d["indx"])
        if want_pos:
            pos.append(d["pos"])
        if want_vel:
            vel.append(d["vel"])
    out = {"indx": np.concatenate(idx)}
    if want_pos:
        out["pos"] = np.concatenate(pos)
    if want_vel:
        out["vel"] = np.concatenate(vel)
    if sort_by_index:
        order = np.argsort(out["indx"], kind="stable")
        for k in ("indx", "pos", "vel"):
            if k in out:
                out[k] = out[k][order]
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="peek a GOTPM snapshot with the reader")
    ap.add_argument("path", help="one subfile, OR a cosmology dir (with --step)")
    ap.add_argument("--step", type=int, help="snapshot step (if path is a dir)")
    ap.add_argument("--prefix", default="INITIAL")
    ap.add_argument("--max-sub", type=int, default=1)
    args = ap.parse_args()

    if args.step is not None:
        print("steps in dir:", snapshot_steps(args.path, args.prefix))
        print("subfiles for step:", len(list_subfiles(args.path, args.step, args.prefix)))
        d = read_snapshot(args.path, args.step, args.prefix, want_vel=True, max_sub=args.max_sub)
    else:
        d = read_subfile(args.path, want_vel=True)

    idx = d["indx"]
    print(f"particles read: {idx.size:,}")
    print(f"index range: {idx.min()} .. {idx.max()}  (2048^3 = {2048**3})")
    print(f"pos min/max per axis:\n{d['pos'].min(0)}\n{d['pos'].max(0)}  cMpc/h")
    print(f"first 3 positions:\n{d['pos'][:3]}")
    print(f"raw velocity abs max: {np.abs(d['vel']).max():.6g}  (code units)")
