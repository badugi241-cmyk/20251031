import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

TICKERS = ["005930.KS", "AAPL", "NVDA"]
STEP = 0.01
TRADING_DAYS = 252

@dataclass
class PortfolioPoint:
    weights: Tuple[float, float, float]
    exp_return: float
    volatility: float
    sharpe: float


def read_prices(path: str) -> List[Dict[str, float]]:
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 4:
        raise ValueError("CSV structure is not as expected (needs header + data rows)")

    metric_row = rows[0]
    ticker_row = rows[1]
    data_rows = rows[3:]

    close_indices: Dict[str, int] = {}
    for idx in range(1, len(metric_row)):
        metric = metric_row[idx].strip()
        ticker = ticker_row[idx].strip()
        if metric == "Close" and ticker in TICKERS:
            close_indices[ticker] = idx

    missing = [ticker for ticker in TICKERS if ticker not in close_indices]
    if missing:
        raise ValueError(f"Close price columns not found for tickers: {missing}")

    aligned_rows: List[Dict[str, float]] = []
    for row in data_rows:
        if not row or not row[0]:
            continue
        entry: Dict[str, float] = {"Date": row[0]}
        skip = False
        for ticker, idx in close_indices.items():
            value = row[idx].strip()
            if not value:
                skip = True
                break
            entry[ticker] = float(value)
        if not skip:
            aligned_rows.append(entry)

    if len(aligned_rows) < 2:
        raise ValueError("Not enough data rows with complete price information")

    return aligned_rows


def compute_returns(price_rows: List[Dict[str, float]]) -> Dict[str, List[float]]:
    returns: Dict[str, List[float]] = {ticker: [] for ticker in TICKERS}
    for prev, curr in zip(price_rows, price_rows[1:]):
        for ticker in TICKERS:
            prev_price = prev[ticker]
            curr_price = curr[ticker]
            ret = (curr_price / prev_price) - 1.0
            returns[ticker].append(ret)
    return returns


def mean(data: List[float]) -> float:
    if not data:
        raise ValueError("Cannot compute mean of empty list")
    return sum(data) / len(data)


def covariance(series_a: List[float], series_b: List[float]) -> float:
    if len(series_a) != len(series_b):
        raise ValueError("Series must be the same length for covariance")
    n = len(series_a)
    if n < 2:
        raise ValueError("Need at least two observations for covariance")
    mean_a = mean(series_a)
    mean_b = mean(series_b)
    return sum((a - mean_a) * (b - mean_b) for a, b in zip(series_a, series_b)) / (n - 1)


def build_covariance_matrix(returns: Dict[str, List[float]]) -> List[List[float]]:
    matrix: List[List[float]] = []
    for ticker_a in TICKERS:
        row: List[float] = []
        for ticker_b in TICKERS:
            cov = covariance(returns[ticker_a], returns[ticker_b])
            row.append(cov)
        matrix.append(row)
    return matrix


def annualize_daily_return(daily_return: float) -> float:
    return (1.0 + daily_return) ** TRADING_DAYS - 1.0


def annualize_returns(daily_returns: Dict[str, List[float]]) -> Dict[str, float]:
    return {ticker: annualize_daily_return(mean(vals)) for ticker, vals in daily_returns.items()}


def annualize_covariance(daily_cov: List[List[float]]) -> List[List[float]]:
    return [[cell * TRADING_DAYS for cell in row] for row in daily_cov]


def portfolio_performance(weights: Tuple[float, float, float],
                          exp_returns: Dict[str, float],
                          cov_matrix: List[List[float]]) -> Tuple[float, float]:
    exp_return = sum(weights[i] * exp_returns[TICKERS[i]] for i in range(3))
    variance = 0.0
    for i in range(3):
        for j in range(3):
            variance += weights[i] * weights[j] * cov_matrix[i][j]
    volatility = math.sqrt(variance)
    return exp_return, volatility


def generate_portfolios(exp_returns: Dict[str, float],
                        cov_matrix: List[List[float]]) -> List[PortfolioPoint]:
    portfolios: List[PortfolioPoint] = []
    steps = int(1 / STEP)
    for i in range(steps + 1):
        w1 = i * STEP
        for j in range(steps + 1 - i):
            w2 = j * STEP
            w3 = 1.0 - w1 - w2
            weights = (round(w1, 4), round(w2, 4), round(w3, 4))
            exp_return, volatility = portfolio_performance(weights, exp_returns, cov_matrix)
            sharpe = exp_return / volatility if volatility > 0 else float("inf")
            portfolios.append(PortfolioPoint(weights, exp_return, volatility, sharpe))
    return portfolios


def efficient_frontier(portfolios: List[PortfolioPoint]) -> List[PortfolioPoint]:
    sorted_by_risk = sorted(portfolios, key=lambda p: p.volatility)
    frontier: List[PortfolioPoint] = []
    best_return = float("-inf")
    for point in sorted_by_risk:
        if point.exp_return > best_return:
            frontier.append(point)
            best_return = point.exp_return
    return frontier


def svg_point(x: float, y: float) -> str:
    return f"{x:.2f},{y:.2f}"


def generate_svg(portfolios: List[PortfolioPoint],
                 frontier: List[PortfolioPoint],
                 min_var: PortfolioPoint,
                 max_sharpe: PortfolioPoint,
                 path: str) -> None:
    width, height = 800, 520
    margin_left, margin_bottom, margin_right, margin_top = 80, 60, 40, 40

    risks = [p.volatility for p in portfolios]
    returns = [p.exp_return for p in portfolios]
    min_risk, max_risk = min(risks), max(risks)
    min_return, max_return = min(returns), max(returns)

    def scale_x(risk: float) -> float:
        if max_risk == min_risk:
            return margin_left
        return margin_left + (risk - min_risk) / (max_risk - min_risk) * (width - margin_left - margin_right)

    def scale_y(ret: float) -> float:
        if max_return == min_return:
            return height - margin_bottom
        # SVG y increases downward, so invert
        return height - margin_bottom - (ret - min_return) / (max_return - min_return) * (height - margin_top - margin_bottom)

    def axis_ticks(start: float, end: float, count: int) -> List[float]:
        if count <= 1:
            return [start]
        step = (end - start) / (count - 1)
        return [start + i * step for i in range(count)]

    x_ticks = axis_ticks(min_risk, max_risk, 5)
    y_ticks = axis_ticks(min_return, max_return, 5)

    svg_lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "  <style>text { font-family: 'Arial', sans-serif; font-size: 14px; }</style>",
        "  <rect x='0' y='0' width='100%' height='100%' fill='white' stroke='none' />",
        "  <g stroke='black' stroke-width='2'>",
        f"    <line x1='{margin_left}' y1='{height - margin_bottom}' x2='{width - margin_right}' y2='{height - margin_bottom}' />",
        f"    <line x1='{margin_left}' y1='{height - margin_bottom}' x2='{margin_left}' y2='{margin_top}' />",
        "  </g>",
    ]

    for tick in x_ticks:
        x = scale_x(tick)
        y = height - margin_bottom
        svg_lines.append(f"  <line x1='{x:.2f}' y1='{y}' x2='{x:.2f}' y2='{y + 6}' stroke='black' stroke-width='1' />")
        svg_lines.append(f"  <text x='{x:.2f}' y='{y + 24}' text-anchor='middle'>{tick*100:.2f}%</text>")

    for tick in y_ticks:
        x = margin_left
        y = scale_y(tick)
        svg_lines.append(f"  <line x1='{x - 6}' y1='{y:.2f}' x2='{x}' y2='{y:.2f}' stroke='black' stroke-width='1' />")
        svg_lines.append(f"  <text x='{x - 10}' y='{y + 5:.2f}' text-anchor='end'>{tick*100:.2f}%</text>")

    svg_lines.append("  <text x='400' y='500' text-anchor='middle'>연간 변동성</text>")
    svg_lines.append(f"  <text x='20' y='{margin_top}' text-anchor='start' transform='rotate(-90, 20, {margin_top})'>연간 기대수익률</text>")
    svg_lines.append("  <g fill='rgba(30, 136, 229, 0.35)' stroke='none'>")
    for point in portfolios:
        x = scale_x(point.volatility)
        y = scale_y(point.exp_return)
        svg_lines.append(f"    <circle cx='{x:.2f}' cy='{y:.2f}' r='3' />")
    svg_lines.append("  </g>")

    svg_lines.append("  <polyline fill='none' stroke='#d32f2f' stroke-width='3' points='" +
                    " ".join(svg_point(scale_x(p.volatility), scale_y(p.exp_return)) for p in frontier) + "' />")

    for label, color, point in (("최소분산", "#388e3c", min_var), ("최대 샤프", "#fbc02d", max_sharpe)):
        x = scale_x(point.volatility)
        y = scale_y(point.exp_return)
        svg_lines.append(f"  <circle cx='{x:.2f}' cy='{y:.2f}' r='6' fill='{color}' stroke='black' stroke-width='1.5' />")
        svg_lines.append(f"  <text x='{x + 10:.2f}' y='{y - 10:.2f}'>{label}</text>")

    svg_lines.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def main() -> None:
    price_rows = read_prices("temp.csv")
    daily_returns = compute_returns(price_rows)
    annual_returns = annualize_returns(daily_returns)
    daily_cov = build_covariance_matrix(daily_returns)
    annual_cov = annualize_covariance(daily_cov)

    portfolios = generate_portfolios(annual_returns, annual_cov)
    frontier = efficient_frontier(portfolios)
    min_var = min(portfolios, key=lambda p: p.volatility)
    max_sharpe = max(portfolios, key=lambda p: p.sharpe)

    generate_svg(portfolios, frontier, min_var, max_sharpe, "efficient_frontier.svg")

    print("Annualized mean returns (%):")
    for ticker in TICKERS:
        print(f"  {ticker}: {annual_returns[ticker] * 100:.2f}%")

    print("\nMin-variance portfolio:")
    print(f"  weights={min_var.weights}, return={min_var.exp_return*100:.2f}%, volatility={min_var.volatility*100:.2f}%, sharpe={min_var.sharpe:.2f}")
    print("\nMax-sharpe portfolio:")
    print(f"  weights={max_sharpe.weights}, return={max_sharpe.exp_return*100:.2f}%, volatility={max_sharpe.volatility*100:.2f}%, sharpe={max_sharpe.sharpe:.2f}")


if __name__ == "__main__":
    main()
