"""Validate a real GOTPM/Multiverse snapshot file before wiring up the full reader.

Reads the ASCII header, then decodes the first few particle records under BOTH
byte orders and checks which one yields sane absolute positions (in [0, boxsize])
and particle indices (in [0, Np)). This settles the two things Dr. Hong warned
about -- header size and endianness -- against an actual file.

Run on grammar (after VPN):
    python -m mvinterp.inspect_snapshot /gpfs/mhee7173_snu/Multiverse_CPL/<a_snapshot_file>

No CuPy needed (pure stdlib + numpy).
"""
import sys
import struct
import argparse

CLOSING = "#End of Ascii Header"


def read_header(path):
    """Return (params dict, byte offset where binary particle data begins, raw header)."""
    params, raw = {}, []
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("reached EOF without finding the header closing line")
            raw.append(line)
            text = line.decode("latin-1", "replace").rstrip("\n")
            if text.strip() == CLOSING:
                offset = f.tell()
                break
            if text.startswith("define"):
                body = text[len("define"):]
                if "=" in body:
                    key, val = body.split("=", 1)
                    params[key.strip()] = val.strip()
    return params, offset, b"".join(raw)


def _get(params, key, cast, default=None):
    if key in params:
        try:
            return cast(params[key])
        except ValueError:
            return cast(float(params[key]))
    return default


def decode_records(path, offset, ptypesize, boxsize, nx, mx, mxmy, ngrid, k=5):
    with open(path, "rb") as f:
        f.seek(offset)
        blob = f.read(k * ptypesize)
    got = len(blob) // ptypesize
    print(f"\nread {got} records of {ptypesize} bytes each")
    print(f"(indx is the GLOBAL Lagrangian index, so it ranges 0..{ngrid} across the "
          f"whole 2048^3 box, not just this subfile)\n")

    if ptypesize != 32:
        print(f"  NOTE: record size is {ptypesize}, not the plain 32 (6 float + int64).")
        print("  Compile flags likely differ (MULTIMASS/INDTIME/...). Hex of record 0:")
        print("   ", blob[:ptypesize].hex())
        print("  -> tell me this hex + ptypesize and I'll adapt the record layout.")
        return

    cell = boxsize / nx
    for order, tag in (("<", "little-endian"), (">", "big-endian")):
        print(f"  ===== interpreting as {tag} ({order}) =====")
        ok = True
        for i in range(got):
            rec = blob[i * 32:(i + 1) * 32]
            x, y, z, vx, vy, vz, indx = struct.unpack(order + "6fq", rec)
            # absolute position from in-cell offset + index-derived grid cell
            ax = ((x + (indx % mx)) % nx) * cell
            ay = ((y + ((indx % mxmy) // mx)) % nx) * cell
            az = ((z + (indx // mxmy)) % nx) * cell
            good = (0 <= indx < ngrid) and all(0 <= a <= boxsize for a in (ax, ay, az))
            ok &= good
            flag = "OK " if good else "BAD"
            print(f"    [{flag}] indx={indx:>12d}  offset=({x:+.4f},{y:+.4f},{z:+.4f})"
                  f"  abs=({ax:7.2f},{ay:7.2f},{az:7.2f}) cMpc/h  v=({vx:+.2f},{vy:+.2f},{vz:+.2f})")
        print(f"    -> {tag}: {'LOOKS CORRECT' if ok else 'invalid (wrong byte order?)'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to one snapshot file")
    ap.add_argument("--nspace", type=int, default=1, help="used only if Mx/MxMy absent")
    ap.add_argument("-k", type=int, default=5, help="how many particles to show")
    ap.add_argument("--raw", action="store_true", help="print the full raw ASCII header")
    args = ap.parse_args()

    params, offset, raw = read_header(args.path)
    if args.raw:
        print(raw.decode("latin-1", "replace"))

    boxsize = _get(params, "Boxsize(Mpc/h)", float)
    nx = _get(params, "Nx", int)
    ny = _get(params, "Ny", int, nx)
    nz = _get(params, "Nz", int, nx)
    npart = _get(params, "Np", int)                     # particles in THIS subfile
    ngrid = nx * ny * nz if nx else None                # global index range (2048^3)
    ptypesize = _get(params, "Particle_TYPE_Size", int, 32)
    mx = _get(params, "Mx", int) or (nx // args.nspace if nx else None)
    mxmy = _get(params, "MxMy", int) or (mx * (ny // args.nspace) if mx else None)

    print("=== parsed header ===")
    for k in ("OmegaMatter0", "OmegaLambda0", "Hubble", "Boxsize(Mpc/h)", "Nx",
              "Mx", "MxMy", "Np", "Zmin", "Zmax", "Anow", "Particle_TYPE_Size"):
        if k in params:
            print(f"  {k:20s} = {params[k]}")
    print(f"  header bytes         = {offset}")
    print(f"  (using) boxsize={boxsize}  nx={nx}  mx={mx}  mxmy={mxmy}  "
          f"ngrid={ngrid}  Np(subfile)={npart}  rec={ptypesize}")

    if None in (boxsize, nx, mx, mxmy):
        print("\n  WARNING: some params missing from header; rerun with --raw and share it.")
        return
    decode_records(args.path, offset, ptypesize, boxsize, nx, mx, mxmy, ngrid, args.k)


if __name__ == "__main__":
    main()
