// ====== 0. Theme toggle ======
        (function() {
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

            document.addEventListener('DOMContentLoaded', function() {
                var btn = document.getElementById('theme-toggle');
                if (!btn) return;
                updateIcon();
                btn.addEventListener('click', function() {
                    var next = isDark() ? 'light' : 'dark';
                    document.documentElement.setAttribute('data-theme', next);
                    localStorage.setItem('theme', next);
                    updateIcon();
                });
            });

            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
                if (!document.documentElement.getAttribute('data-theme')) updateIcon();
            });
        })();

// ====== 1. Generate full-page grain/noise texture ======
        (function initGrain() {
            var gc = document.getElementById('grain-overlay');
            var W = 256, H = 256;
            gc.width = W;
            gc.height = H;
            gc.style.width = '100%';
            gc.style.height = '100%';
            gc.style.imageRendering = 'pixelated';

            var gctx = gc.getContext('2d');
            var img = gctx.createImageData(W, H);
            var d = img.data;

            function renderGrain() {
                for (var i = 0; i < d.length; i += 4) {
                    var v = Math.random() * 255;
                    // Tint the noise toward red/warm tones
                    d[i]     = v;                      // R — full
                    d[i + 1] = v * 0.3;                // G — reduced
                    d[i + 2] = v * 0.1;                // B — minimal
                    d[i + 3] = 255;
                }
                gctx.putImageData(img, 0, 0);
            }

            renderGrain();
            // Re-randomize grain subtly every 100ms for a living thermal feel
            setInterval(renderGrain, 120);
        })();

        // ====== 2. Red thermal palette + grain for profile image ======
        var THERMAL_PALETTE = [
            [0,   0,   0],
            [25,  0,   0],
            [55,  0,   0],
            [90,  0,   0],
            [130, 0,   0],
            [170, 5,   0],
            [200, 20,  0],
            [230, 45,  0],
            [255, 80,  0],
            [255, 130, 5],
            [255, 185, 40],
            [255, 225, 100],
            [255, 248, 190],
        ];

        function lerp(a, b, t) { return a + (b - a) * t; }

        function brightnessToThermal(brightness) {
            var idx = brightness * (THERMAL_PALETTE.length - 1);
            var lo  = Math.floor(idx);
            var hi  = Math.min(lo + 1, THERMAL_PALETTE.length - 1);
            var t   = idx - lo;
            return [
                lerp(THERMAL_PALETTE[lo][0], THERMAL_PALETTE[hi][0], t),
                lerp(THERMAL_PALETTE[lo][1], THERMAL_PALETTE[hi][1], t),
                lerp(THERMAL_PALETTE[lo][2], THERMAL_PALETTE[hi][2], t),
            ];
        }

        function applyThermalEffect() {
            var img    = document.getElementById('profile-img');
            var canvas = document.getElementById('thermal-canvas');

            try {
                var ctx = canvas.getContext('2d');
                canvas.width  = img.naturalWidth;
                canvas.height = img.naturalHeight;
                ctx.drawImage(img, 0, 0);

                var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                var d = imageData.data;
                var w = canvas.width;

                for (var i = 0; i < d.length; i += 4) {
                    var gray = (d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114) / 255;

                    // Add grain noise to the brightness before mapping
                    var noise = (Math.random() - 0.5) * 0.08;
                    gray = Math.max(0, Math.min(1, gray + noise));

                    var rgb  = brightnessToThermal(gray);
                    d[i]     = rgb[0];
                    d[i + 1] = rgb[1];
                    d[i + 2] = rgb[2];
                }

                ctx.putImageData(imageData, 0, 0);

                // Slight gaussian-ish softening to mimic thermal sensor resolution
                ctx.globalAlpha = 0.15;
                ctx.filter = 'blur(1px)';
                ctx.drawImage(canvas, 0, 0);
                ctx.globalAlpha = 1.0;
                ctx.filter = 'none';

                canvas.style.display = 'block';
                img.style.display    = 'none';
            } catch (e) {
                img.classList.add('thermal-fallback');
                canvas.style.display = 'none';
                img.style.display    = 'block';
            }
        }

        var profileImg = document.getElementById('profile-img');
        if (profileImg.complete && profileImg.naturalWidth > 0) {
            applyThermalEffect();
        } else {
            profileImg.onload = applyThermalEffect;
            profileImg.onerror = function() {
                profileImg.classList.add('thermal-fallback');
            };
        }

        // ====== 3. SPA routing ======
        function handleRouting() {
            var hash = window.location.hash || '#home';

            // Docs subroutes: #docs and #docs/... all map to docs-page
            var pageHash = hash.startsWith('#docs') ? '#docs' : hash;
            var targetId = pageHash.replace('#', '') + '-page';

            document.querySelectorAll('.page-container').forEach(function(page) {
                page.classList.remove('active');
            });
            document.querySelectorAll('.nav-link').forEach(function(link) {
                link.classList.remove('active');
            });

            var targetPage = document.getElementById(targetId);
            if (targetPage) {
                targetPage.classList.add('active');
                var activeNav = document.querySelector('.nav-link[href="' + pageHash + '"]');
                if (activeNav) activeNav.classList.add('active');
                if (pageHash === '#docs') loadDocsList();
            } else {
                document.getElementById('home-page').classList.add('active');
            }

            // Show sidebar toggle only on docs page
            var sidebarBtn = document.getElementById('docs-sidebar-toggle');
            if (sidebarBtn) {
                sidebarBtn.style.display = (pageHash === '#docs') ? 'flex' : 'none';
            }
        }

        // ====== 4. Mika Doc ======
        var DOCS_GITHUB_BASE = 'https://github.com/MikaStars39/Mika_Playbook/blob/main/';
        var currentDocRaw = null;

        function renderMarkdown(md) {
            var blocks = [];
            md = md.replace(/\$\$([\s\S]*?)\$\$/g, function(_, math) {
                blocks.push({ type: 'display', math: math });
                return '\x02MATH' + (blocks.length - 1) + '\x03';
            });
            md = md.replace(/\$([^\$\n]+?)\$/g, function(_, math) {
                blocks.push({ type: 'inline', math: math });
                return '\x02MATH' + (blocks.length - 1) + '\x03';
            });
            var html = marked.parse(md);
            html = html.replace(/\x02MATH(\d+)\x03/g, function(_, i) {
                var b = blocks[+i];
                try {
                    return katex.renderToString(b.math, { displayMode: b.type === 'display', throwOnError: false });
                } catch(e) { return b.math; }
            });
            return html;
        }

        // Resolve a relative link against a base doc path
        function resolvePath(base, rel) {
            var parts = base.split('/');
            parts.pop();
            rel.split('/').forEach(function(seg) {
                if (seg === '..') parts.pop();
                else if (seg !== '.') parts.push(seg);
            });
            return parts.join('/');
        }

        function interceptDocLinks(currentPath) {
            document.querySelectorAll('#docs-body a').forEach(function(a) {
                var href = a.getAttribute('href');
                if (!href || /^(https?:|mailto:|#)/.test(href)) return;
                if (href.endsWith('.md')) {
                    a.addEventListener('click', function(e) {
                        e.preventDefault();
                        var resolved = resolvePath(currentPath, href);
                        loadDoc(resolved, document.querySelectorAll('#docs-list a'));
                    });
                }
            });
        }

        function renderToolbar(path) {
            var toolbar = document.getElementById('docs-toolbar');
            toolbar.innerHTML = '';
            toolbar.classList.remove('welcome-mode');

            var actions = document.createElement('div');
            actions.className = 'docs-toolbar-actions';

            var copyBtn = document.createElement('button');
            copyBtn.className = 'docs-toolbar-btn';
            copyBtn.textContent = 'Copy Markdown';
            copyBtn.onclick = function() {
                navigator.clipboard.writeText(currentDocRaw || '').then(function() {
                    copyBtn.textContent = 'Copied!';
                    setTimeout(function() { copyBtn.textContent = 'Copy Markdown'; }, 2000);
                });
            };

            var sourceLink = document.createElement('a');
            sourceLink.className = 'docs-toolbar-btn';
            sourceLink.href = DOCS_GITHUB_BASE + path;
            sourceLink.target = '_blank';
            sourceLink.rel = 'noopener';
            sourceLink.textContent = 'View Source';

            actions.appendChild(copyBtn);
            actions.appendChild(sourceLink);
            toolbar.appendChild(actions);
        }

        function showDocsWelcome(files) {
            var toolbar = document.getElementById('docs-toolbar');
            toolbar.innerHTML = '';
            toolbar.classList.add('welcome-mode');
            renderCollapseBtn(toolbar);

            var quickLinks = files.slice(0, 5).map(function(f) {
                return '<a class="docs-welcome-link" href="#docs/' + encodeURIComponent(f.path) + '">→ ' + f.title + '</a>';
            }).join('');
            document.getElementById('docs-body').innerHTML =
                '<div class="docs-welcome">' +
                '<h2>Mika\'s Knowledge Base</h2>' +
                '<p>A collection of notes on machine learning, systems, and ideas. Select a topic from the sidebar to explore.</p>' +
                '<div class="docs-welcome-links">' + quickLinks + '</div>' +
                '</div>';
        }

        function loadDocsList() {
            fetch('docs/manifest.json')
                .then(function(r) { return r.json(); })
                .then(function(files) {
                    var list = document.getElementById('docs-list');
                    list.innerHTML = '';

                    var groups = {};
                    var groupOrder = [];
                    files.forEach(function(f) {
                        var g = f.group || '';
                        if (!groups[g]) { groups[g] = []; groupOrder.push(g); }
                        groups[g].push(f);
                    });

                    groupOrder.forEach(function(g) {
                        if (g) {
                            var details = document.createElement('details');
                            details.open = true;
                            var summary = document.createElement('summary');
                            summary.textContent = g;
                            details.appendChild(summary);
                            groups[g].forEach(function(f) { details.appendChild(makeDocLink(f)); });
                            list.appendChild(details);
                        } else {
                            groups[g].forEach(function(f) { list.appendChild(makeDocLink(f)); });
                        }
                    });

                    var hash = window.location.hash;
                    if (hash.startsWith('#docs/')) {
                        loadDoc(decodeURIComponent(hash.slice(6)), list.querySelectorAll('a'));
                    } else {
                        showDocsWelcome(files);
                    }
                })
                .catch(function() {
                    document.getElementById('docs-body').innerHTML =
                        '<p style="color:var(--text-muted)">文档加载失败，请稍后再试。</p>';
                });
        }

        function makeDocLink(f) {
            var a = document.createElement('a');
            a.href = '#docs/' + encodeURIComponent(f.path);
            a.textContent = f.title;
            return a;
        }

        function loadDoc(path, links) {
            if (links) {
                links.forEach(function(a) {
                    a.classList.toggle('active',
                        decodeURIComponent(a.href.split('#docs/')[1] || '') === path);
                });
            }
            fetch('docs/' + path)
                .then(function(r) { return r.text(); })
                .then(function(md) {
                    currentDocRaw = md;
                    document.getElementById('docs-body').innerHTML = renderMarkdown(md);
                    renderToolbar(path);
                    interceptDocLinks(path);
                });
            history.replaceState(null, '', '#docs/' + encodeURIComponent(path));
        }

        window.addEventListener('hashchange', handleRouting);
        window.addEventListener('DOMContentLoaded', handleRouting);

        // ====== 5. Sidebar toggle (nav button) ======
        (function() {
            var toggle  = document.getElementById('docs-sidebar-toggle');
            var sidebar = document.querySelector('.docs-sidebar');
            var overlay = document.getElementById('docs-sidebar-overlay');
            if (!toggle || !sidebar || !overlay) return;

            function openSidebar() {
                sidebar.classList.add('open');
                overlay.classList.add('open');
            }

            function closeSidebar() {
                sidebar.classList.remove('open');
                overlay.classList.remove('open');
            }

            toggle.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    // Mobile: slide-in drawer
                    sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
                } else {
                    // Desktop: collapse / expand sidebar in layout
                    var layout = document.querySelector('.docs-layout');
                    if (layout) layout.classList.toggle('sidebar-collapsed');
                }
            });

            overlay.addEventListener('click', closeSidebar);

            // Close drawer when a doc link is clicked on mobile
            sidebar.addEventListener('click', function(e) {
                if (e.target.tagName === 'A') closeSidebar();
            });
        })();
