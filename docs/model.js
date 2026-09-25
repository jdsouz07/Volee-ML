// model.js — the demo's forecasting arithmetic, the browser twin of
// volee_ml/export.py (demo_features + demo_probability). Keep them in step:
// tests/test_export.py runs this file under Node and checks it matches.

const SCALE = 173.7178;

function g(phi) {
  return 1 / Math.sqrt(1 + (3 * phi * phi) / (Math.PI * Math.PI));
}

// Glicko-2 forecast that A beats B (glicko2.py: win_probability).
function winProbability(a, b) {
  const muA = (a[0] - 1500) / SCALE, muB = (b[0] - 1500) / SCALE;
  const phi = Math.sqrt((a[1] / SCALE) ** 2 + (b[1] / SCALE) ** 2);
  return 1 / (1 + Math.exp(-g(phi) * (muA - muB)));
}

// Rebuild every model feature for A vs B, exactly as features.py defines it.
function demoFeatures(data, aId, bId) {
  const a = data.players[aId], b = data.players[bId];
  const aWins = data.h2h[`${aId}|${bId}`] || 0;
  const bWins = data.h2h[`${bId}|${aId}`] || 0;
  return {
    glicko_prob: winProbability(a.volee, b.volee),
    margin_prob: winProbability(a.margin, b.margin),
    rating_diff: a.volee[0] - b.volee[0],
    rd_a: a.volee[1],
    rd_b: b.volee[1],
    experience_diff: Math.log1p(a.matches) - Math.log1p(b.matches),
    rest_days_a: a.rest,
    rest_days_b: b.rest,
    form_diff: a.form - b.form,
    games_share_diff: a.games_share - b.games_share,
    momentum_diff: a.momentum - b.momentum,
    h2h_edge: (aWins - bWins) / (aWins + bWins + 2),
    h2h_meetings: Math.log1p(aWins + bWins),
    age_a: a.age,
    age_b: b.age,
    is_womens: a.tour === "wta" ? 1 : 0,
  };
}

// Logistic regression: scale each feature, weight it, add up, squash.
// Also returns each feature's contribution, for the "why" list on the page.
function demoProbability(data, features) {
  const m = data.model;
  let total = m.intercept;
  const parts = [];
  m.features.forEach((name, i) => {
    let x = features[name];
    if (m.logit_features.includes(name)) {
      const p = Math.min(Math.max(x, 1e-6), 1 - 1e-6);
      x = Math.log(p / (1 - p));
    }
    const contribution = (m.weights[i] * (x - m.mean[i])) / m.scale[i];
    total += contribution;
    parts.push({ name, contribution });
  });
  return { probability: 1 / (1 + Math.exp(-total)), parts };
}

if (typeof module !== "undefined") {
  module.exports = { winProbability, demoFeatures, demoProbability };
}
