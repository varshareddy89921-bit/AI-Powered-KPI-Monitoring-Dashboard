from flask import Flask, jsonify, render_template, request
import kpi_engine as ke

app = Flask(__name__)


def _list(name, legacy=None):
    v = request.args.get(name) or (request.args.get(legacy) if legacy else "") or ""
    return [x for x in v.split(",") if x and x != "All"]


def _num(name):
    try:
        return float(request.args[name])
    except (KeyError, ValueError):
        return None


@app.route("/")
def index():
    return render_template("index.html", meta=ke.meta())


@app.route("/api/dashboard")
def api():
    """Slicer params (all optional, comma-separated lists): regions, categories, pairs (Region|Category),
    year, start / end (YYYY-MM), rmin / rmax. Legacy ?region= and ?category= still work."""
    return jsonify(ke.dashboard(
        regions=_list("regions", "region"), categories=_list("categories", "category"), pairs=_list("pairs"),
        year=request.args.get("year") if request.args.get("year", "All") != "All" else None,
        start=request.args.get("start") or None, end=request.args.get("end") or None,
        rmin=_num("rmin"), rmax=_num("rmax")))


if __name__ == "__main__":
    app.run(debug=True)
