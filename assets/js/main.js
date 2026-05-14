// ====== 0. Theme toggle ======
(function () {
    var SUN = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
    var MOON = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

    function isDark() {
        var t = document.documentElement.getAttribute('data-theme');
        if (t) return t === 'dark';
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    function updateIcon() {
        var btn = document.getElementById('theme-toggle');
        if (btn) btn.innerHTML = isDark() ? SUN : MOON;
    }

    document.addEventListener('DOMContentLoaded', function () {
        var btn = document.getElementById('theme-toggle');
        if (!btn) return;
        updateIcon();
        btn.addEventListener('click', function () {
            var next = isDark() ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('theme', next);
            updateIcon();
            var r = window.__liquidGLRenderer__;
            if (!r) return;
            r.canvas.style.opacity = '0';
            r._capturing = false;
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    r._capturing = false;
                    r.captureSnapshot().then(function () {
                        r.canvas.style.opacity = '1';
                    });
                });
            });
        });
    });

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () {
        if (!document.documentElement.getAttribute('data-theme')) {
            updateIcon();
            var r = window.__liquidGLRenderer__;
            if (!r) return;
            r.canvas.style.opacity = '0';
            r._capturing = false;
            requestAnimationFrame(function () {
                requestAnimationFrame(function () {
                    r._capturing = false;
                    r.captureSnapshot().then(function () {
                        r.canvas.style.opacity = '1';
                    });
                });
            });
        }
    });
})();

// ====== 1. Mouse-reactive Squares Background ======
(function () {
    var CELL = 18;
    var GAP = 2;
    var RADIUS = 8;
    var DECAY = 0.92;

    var canvas, ctx, w, h, cols, rows;
    var heat;
    var mouseCol = -100, mouseRow = -100;

    function isDark() {
        var t = document.documentElement.getAttribute('data-theme');
        if (t) return t === 'dark';
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    function init() {
        canvas = document.getElementById('snake-bg');
        if (!canvas) return;
        ctx = canvas.getContext('2d');
        resize();
        window.addEventListener('resize', resize);
        document.addEventListener('mousemove', function (e) {
            mouseCol = e.clientX / CELL;
            mouseRow = e.clientY / CELL;
        });
        document.addEventListener('mouseleave', function () {
            mouseCol = -100;
            mouseRow = -100;
        });
        requestAnimationFrame(loop);
    }

    function resize() {
        var dpr = window.devicePixelRatio || 1;
        w = window.innerWidth;
        h = window.innerHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        var newCols = Math.floor(w / CELL);
        var newRows = Math.floor(h / CELL);
        if (newCols !== cols || newRows !== rows) {
            cols = newCols;
            rows = newRows;
            heat = new Float32Array(cols * rows);
        }
    }

    function loop() {
        update();
        draw();
        requestAnimationFrame(loop);
    }

    function update() {
        for (var i = 0; i < heat.length; i++) {
            heat[i] *= DECAY;
        }

        var r0 = Math.max(0, Math.floor(mouseRow - RADIUS));
        var r1 = Math.min(rows - 1, Math.ceil(mouseRow + RADIUS));
        var c0 = Math.max(0, Math.floor(mouseCol - RADIUS));
        var c1 = Math.min(cols - 1, Math.ceil(mouseCol + RADIUS));

        for (var r = r0; r <= r1; r++) {
            for (var c = c0; c <= c1; c++) {
                var dx = c + 0.5 - mouseCol;
                var dy = r + 0.5 - mouseRow;
                var d = Math.sqrt(dx * dx + dy * dy);
                if (d < RADIUS) {
                    var add = 0.15 * (1 - d / RADIUS);
                    var idx = r * cols + c;
                    if (heat[idx] < add) heat[idx] = add;
                }
            }
        }
    }

    function draw() {
        var dark = isDark();
        var bg = dark ? '#111111' : '#fafaf7';
        var sc = dark ? [46, 170, 130] : [13, 92, 71];

        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);

        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                var v = heat[r * cols + c];
                if (v < 0.005) continue;
                var alpha = v * 0.35;
                ctx.fillStyle = 'rgba(' + sc[0] + ',' + sc[1] + ',' + sc[2] + ',' + alpha + ')';
                ctx.fillRect(c * CELL + GAP, r * CELL + GAP, CELL - GAP * 2, CELL - GAP * 2);
            }
        }
    }

    window.__snakeBg__ = function() { return { heat: heat, cols: cols, rows: rows, CELL: CELL }; };

    document.addEventListener('DOMContentLoaded', init);
})();

// ====== Flip ripple ======
function triggerFlipRipple(el) {
    var bg = window.__snakeBg__ && window.__snakeBg__();
    if (!bg || !bg.heat) return;
    var rect = el.getBoundingClientRect();
    var cx = (rect.left + rect.width / 2) / bg.CELL;
    var cy = (rect.top + rect.height / 2) / bg.CELL;
    var maxR = 18;
    var frame = 0;
    var duration = 40;
    function step() {
        if (frame > duration) return;
        var radius = (frame / duration) * maxR;
        var thickness = 3;
        var r0 = Math.max(0, Math.floor(cy - radius - thickness));
        var r1 = Math.min(bg.rows - 1, Math.ceil(cy + radius + thickness));
        var c0 = Math.max(0, Math.floor(cx - radius - thickness));
        var c1 = Math.min(bg.cols - 1, Math.ceil(cx + radius + thickness));
        for (var r = r0; r <= r1; r++) {
            for (var c = c0; c <= c1; c++) {
                var dx = c + 0.5 - cx;
                var dy = r + 0.5 - cy;
                var d = Math.sqrt(dx * dx + dy * dy);
                var diff = Math.abs(d - radius);
                if (diff < thickness) {
                    var intensity = 0.5 * (1 - diff / thickness) * (1 - frame / duration);
                    var idx = r * bg.cols + c;
                    if (bg.heat[idx] < intensity) bg.heat[idx] = intensity;
                }
            }
        }
        frame++;
        requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ====== 2. Initialize liquidGL ======
document.addEventListener('DOMContentLoaded', function () {
    if (typeof liquidGL === 'undefined') return;

    setTimeout(function () {
        liquidGL({
            target: '.glass-target',
            refraction: 0.005,
            bevelDepth: 0.06,
            bevelWidth: 0.2,
            frost: 2,
            shadow: false,
            specular: true,
            tilt: true,
            tiltFactor: 2,
            reveal: 'fade',
        });
    }, 300);
});
