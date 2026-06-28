"""Phase 9: Basic Math, Constants, I/O, Cache, and Magic Functions."""

from axiompy import Axiom

print("=== Basic Math ===")
print(f"5 + 3 = {Axiom.math.add(5, 3)}")
print(f"10 - 4 = {Axiom.math.sub(10, 4)}")
print(f"6 * 7 = {Axiom.math.mul(6, 7)}")
print(f"10 / 3 = {Axiom.math.div(10, 3):.4f}")
print(f"2^10 = {Axiom.math.power(2, 10)}")
print(f"sqrt(144) = {Axiom.math.sqrt(144)}")
print(f"5! = {Axiom.math.factorial(5)}")
print(f"Is 17 prime? {Axiom.math.is_prime(17)}")
print(f"25% of 200 = {Axiom.math.percentage(25, 200)}")

print("\n=== Fractions ===")
f1 = Axiom.Fraction(1, 3)
f2 = Axiom.Fraction(1, 6)
print(f"1/3 + 1/6 = {f1 + f2}")

print("\n=== Constants ===")
print(f"PI = {Axiom.constants.PI}")
print(f"E = {Axiom.constants.E}")
print(f"C = {Axiom.constants.C} m/s")
print(f"G = {Axiom.constants.G} N·m²/kg²")
print(f"All constants: {Axiom.constants.list_all()}")

print("\n=== Data I/O ===")
Axiom.io.save_json({"vector": [1, 2, 3]}, "/tmp/axiompy_demo.json")
loaded = Axiom.io.load_json("/tmp/axiompy_demo.json")
print(f"JSON roundtrip: {loaded}")

Axiom.io.save_csv([["x", "y"], [1, 2], [3, 4]], "/tmp/axiompy_demo.csv", headers=["Name", "Val"])
data = Axiom.io.load("/tmp/axiompy_demo.csv")
print(f"CSV loaded: {len(data)} rows")

Axiom.io.save([0.577, 1.618, 3.141], "/tmp/axiompy_demo.axi")
restored = Axiom.io.load("/tmp/axiompy_demo.axi")
print(f".axi roundtrip: {restored}")

print("\n=== Cache ===")
Axiom.cache.put("answer", 42)
print(f"Cached answer: {Axiom.cache.get('answer')}")
Axiom.cache.put("ephemeral", "gone", ttl=1)
Axiom.cache.invalidate("answer")
print(f"After invalidate: {Axiom.cache.get('answer')}")
Axiom.cache.clear()
print(f"After clear, size = {Axiom.cache.size}")

print("\n=== Magic Functions ===")
result = Axiom.magic.pipe(
    5,
    lambda x: x * 2,
    lambda x: x + 1,
    lambda x: x ** 2,
)
print(f"Pipe: {result}")

add1 = lambda x: x + 1
double = lambda x: x * 2
f = Axiom.magic.compose(add1, double)
print(f"Compose: f(5) = {f(5)}")

print("\nAll Phase 9 demonstrations passed!")
