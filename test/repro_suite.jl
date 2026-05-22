using Test, Tarang

prefix = split(get(ENV, "PREFIX", ""), ",", keepempty=false)
for f in prefix
    try
        include(String(f))
    catch e
        println("(include $f raised $(typeof(e)) — continuing)")
    end
end

# ---- measure transform steady-state allocation (mirrors test_transform_inplace) ----
coords = CartesianCoordinates("x", "y")
dist   = Distributor(coords; dtype=Float64, device=CPU())
xb     = RealFourier(coords["x"]; size=128, bounds=(0.0, 2π))
yb     = RealFourier(coords["y"]; size=128, bounds=(0.0, 2π))
domain = Domain(dist, (xb, yb))
T = ScalarField(domain, "T")
pg = Tarang.get_grid_data(T)
for j in 1:128, i in 1:128
    pg[i,j] = sin(2π*(i-1)/128)*cos(2π*(j-1)/128)
end
T.current_layout = :g
for _ in 1:3; ensure_layout!(T, :c); ensure_layout!(T, :g); end
rt = @allocated for _ in 1:100; ensure_layout!(T, :c); ensure_layout!(T, :g); end
println(">>> PREFIX=[", get(ENV, "PREFIX", ""), "]")
println(">>> VERSION=", VERSION, "  round-trip/100=", rt, "  per=", rt/100, "  ", rt < 200_000 ? "PASS" : "FAIL")
