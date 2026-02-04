"""
Shared JavaScript snippets and constants for browser automation tools.
"""

COMMON_CLOSE_SELECTORS = [
    "button[aria-label*='close' i]",
    "button[class*='close' i]",
    "[id*='close' i]",
    "svg[data-icon='close']",
    "text=/^Close$/i",
    "text=/^No thanks$/i",
    "text=/^Not now$/i",
    "text=/^Skip$/i",
    "text=/^X$/"
]

JS_REMOVE_OVERLAYS = """() => {
    let count = 0;
    const elements = document.querySelectorAll('*');
    for (const el of elements) {
        const style = window.getComputedStyle(el);
        if ((style.position === 'fixed' || style.position === 'absolute') && parseInt(style.zIndex) > 50) {
            const rect = el.getBoundingClientRect();
            if (rect.width > window.innerWidth * 0.8 && rect.height > window.innerHeight * 0.8) {
                el.remove();
                count++;
            }
        }
    }
    return count;
}"""

JS_SCROLL_SLOW = """async () => {
    const distance = 800;
    const delay = 100;
    const maxScrolls = 50;
    let scrolls = 0;
    while (document.scrollingElement.scrollTop + window.innerHeight < document.scrollingElement.scrollHeight && scrolls < maxScrolls) {
        document.scrollingElement.scrollBy(0, distance);
        await new Promise(resolve => setTimeout(resolve, delay));
        scrolls++;
    }
}"""

JS_EXTRACT_LINKS = """() => {
    const anchors = Array.from(document.querySelectorAll('a'));
    return anchors.map(a => {
        let context = 'content';
        if (a.closest('nav')) context = 'nav';
        else if (a.closest('header') || a.closest('.header')) context = 'header';
        else if (a.closest('footer') || a.closest('.footer')) context = 'footer';
        else if (a.closest('aside') || a.closest('.sidebar')) context = 'sidebar';
        return {
            text: a.innerText.trim() || a.getAttribute('aria-label') || '',
            href: a.href,
            context: context
        };
    }).filter(link => link.href.startsWith('http') && link.text.length > 0);
}"""

JS_ANALYZE_STRUCTURE = """() => {
    // Helper to check visibility
    const isVisible = (el) => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
    };

    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4')).filter(isVisible).map(h => ({
        tag: h.tagName.toLowerCase(),
        text: h.innerText.trim()
    })).filter(h => h.text.length > 0);

    const interactive = Array.from(document.querySelectorAll('a, button, input, select, textarea, [role="button"]')).filter(isVisible).map(el => ({
        tag: el.tagName.toLowerCase(),
        text: (el.innerText || el.value || el.getAttribute('aria-label') || '').substring(0, 50).replace(/\\s+/g, ' ').trim(),
        id: el.id || null,
        href: el.href || null
    })).filter(i => i.text.length > 0);

    return { headings, interactive: interactive.slice(0, 100) }; // Limit to avoid context overflow
}"""

JS_HIGHLIGHT_ELEMENTS = """(selector) => {
    const elements = document.querySelectorAll(selector);
    elements.forEach(el => {
        el.style.outline = '4px solid red';
        el.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
    });
    return elements.length;
}"""

JS_REMOVE_HIGHLIGHTS = """() => {
    const elements = document.querySelectorAll('*');
    elements.forEach(el => {
        if (el.style.outline.includes('red') || el.style.backgroundColor.includes('rgba(255, 0, 0, 0.1)')) {
            el.style.outline = '';
            el.style.backgroundColor = '';
        }
    });
}"""

JS_GET_COMPUTED_STYLE = """(selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return {
        display: style.display,
        visibility: style.visibility,
        opacity: style.opacity,
        zIndex: style.zIndex,
        position: style.position,
        width: style.width,
        height: style.height,
        pointerEvents: style.pointerEvents,
        overflow: style.overflow,
        is_visible_on_screen: (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none'),
        rect: {
            width: rect.width,
            height: rect.height,
            top: rect.top,
            left: rect.left
        }
    };
}"""

JS_GET_DROPDOWN_OPTIONS = """(selector) => {
    const select = document.querySelector(selector);
    if (!select) return null;
    return Array.from(select.options).map(opt => ({
        text: opt.text,
        value: opt.value,
        selected: opt.selected
    }));
}"""

JS_ENABLE_LOG_CAPTURE = """() => {
    if (!window._captured_logs) {
        window._captured_logs = [];
        const methods = ['log', 'info', 'warn', 'error'];
        methods.forEach(method => {
            const original = console[method];
            console[method] = (...args) => {
                window._captured_logs.push({
                    type: method, 
                    message: args.map(String).join(' '), 
                    timestamp: new Date().toISOString()
                });
                original.apply(console, args);
            };
        });
    }
}"""

JS_FIND_TEXT_ELEMENTS = """(text) => {
    const results = [];
    const all = document.querySelectorAll('*');
    for (const el of all) {
        if (el.offsetParent !== null && el.textContent.includes(text)) {
             let childHasText = false;
             for (const child of el.children) {
                 if (child.textContent.includes(text)) {
                     childHasText = true;
                     break;
                 }
             }
             if (!childHasText) {
                results.push({
                    tag: el.tagName.toLowerCase(),
                    text: el.innerText.trim().substring(0, 50),
                    id: el.id,
                    classes: el.className
                });
             }
        }
    }
    return results.slice(0, 20);
}"""

JS_GET_LOCAL_STORAGE = """() => {
    const items = {};
    try {
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            items[key] = localStorage.getItem(key);
        }
    } catch (e) {
        return { error: 'Access to localStorage denied or failed: ' + e.toString() };
    }
    return items;
}"""

JS_WAIT_FOR_DOM_STABILITY = """async () => {
    return new Promise((resolve) => {
        let timer;
        const observer = new MutationObserver(() => {
            if (timer) clearTimeout(timer);
            timer = setTimeout(() => {
                observer.disconnect();
                resolve('stable');
            }, 500);
        });
        observer.observe(document.body, { childList: true, subtree: true, attributes: true });
        // Max wait 2s to avoid hanging
        setTimeout(() => {
            observer.disconnect();
            resolve('timeout');
        }, 2000);
    });
}"""

JS_REMOVE_ADS = """() => {
    const selectors = [
        'iframe[src*="googleads"]',
        'iframe[src*="doubleclick"]',
        'iframe[src*="amazon-adsystem"]',
        'iframe[src*="adnxs"]',
        'div[id^="div-gpt-ad"]',
        'div[class*="ad-container"]',
        'div[class*="ad_wrapper"]',
        'div[class*="text-ad"]',
        '.adsbygoogle',
        '#ad_unit',
        '.ad-banner',
        '.advertisement',
        '[aria-label="Advertisement"]',
        '[class*="sponsored"]',
        '[id*="sponsored"]',
        '.sticky-ad',
        '.fixed-ad',
        'div[data-ad-unit]',
        '.video-ad',
        '.preroll-ad',
        '#google_vignette',
        'ins.adsbygoogle[data-ad-status="unfilled"]',
        'div[id^="google_ads_iframe"]'
    ];
    let count = 0;
    
    // 1. Remove by selector
    selectors.forEach(sel => {
        document.querySelectorAll(sel).forEach(el => {
            if (el.tagName !== 'BODY' && el.tagName !== 'HTML') {
                el.remove();
                count++;
            }
        });
    });

    // 2. Remove iframes that are likely ads
    document.querySelectorAll('iframe').forEach(iframe => {
        try {
            const src = iframe.src || '';
            if (src.includes('ads') || src.includes('doubleclick') || src.includes('tracking')) {
                 iframe.remove();
                 count++;
            }
        } catch(e) {}
    });

    return count;
}"""

JS_GET_SCROLL_INFO = """() => {
    const doc = document.documentElement;
    const win = window;
    const scrollTop = win.scrollY || doc.scrollTop;
    const scrollHeight = doc.scrollHeight;
    const clientHeight = doc.clientHeight;
    // Allow a small margin of error (e.g. 5px) for "bottom" detection
    const isAtBottom = Math.ceil(scrollTop + clientHeight) >= scrollHeight - 5;
    return {
        isAtBottom,
        percent: scrollHeight > 0 ? Math.round(((scrollTop + clientHeight) / scrollHeight) * 100) : 100
    };
}"""

JS_ASSESS_SECTION = """() => {
    const viewportHeight = window.innerHeight;
    const scrollY = window.scrollY;
    const docHeight = document.documentElement.scrollHeight;
    
    // Check for visible inputs
    const inputs = Array.from(document.querySelectorAll('input:not([type="hidden"]), textarea, select'));
    const visibleInputs = inputs.filter(el => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return rect.top >= 0 && rect.bottom <= viewportHeight && 
               style.visibility !== 'hidden' && style.display !== 'none' && style.opacity !== '0';
    });
    
    const unfilledCount = visibleInputs.filter(el => {
        if (el.type === 'checkbox' || el.type === 'radio') return false; 
        return !el.value;
    }).length;

    // Check for primary action buttons
    const buttons = Array.from(document.querySelectorAll('button, input[type="submit"], a[href], [role="button"]'));
    const visibleButtons = buttons.filter(el => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return rect.top >= 0 && rect.bottom <= viewportHeight && 
               style.visibility !== 'hidden' && style.display !== 'none';
    });
    
    const actionKeywords = ['submit', 'next', 'continue', 'finish', 'complete', 'search', 'login', 'sign in', 'post'];
    const primaryActions = visibleButtons.filter(el => {
        const text = (el.innerText || el.value || '').toLowerCase();
        return actionKeywords.some(kw => text.includes(kw));
    }).map(b => (b.innerText || b.value || 'Button').trim().substring(0, 30));

    const isAtBottom = Math.ceil(scrollY + viewportHeight) >= docHeight - 10;

    return {
        unfilled_inputs: unfilledCount,
        actions: primaryActions,
        at_bottom: isAtBottom,
        progress: docHeight > 0 ? Math.round(((scrollY + viewportHeight) / docHeight) * 100) : 100
    };
}"""

JS_SCROLL_TO_TEXT = """async (text) => {
    const maxScrolls = 60;
    const distance = 600;
    const delay = 100;
    const lowerText = text.toLowerCase();
    
    const findElementWithText = () => {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while (node = walker.nextNode()) {
            if (node.textContent.toLowerCase().includes(lowerText)) {
                const element = node.parentElement;
                // Check if element is visible (has dimensions)
                if (element && (element.offsetWidth > 0 || element.offsetHeight > 0 || element.getClientRects().length > 0)) {
                    return element;
                }
            }
        }
        return null;
    };

    for (let i = 0; i < maxScrolls; i++) {
        const element = findElementWithText();
        if (element) {
            element.scrollIntoView({behavior: 'smooth', block: 'center'});
            return true;
        }
        if ((window.innerHeight + window.scrollY) >= document.documentElement.scrollHeight - 10) {
            break;
        }
        window.scrollBy(0, distance);
        await new Promise(resolve => setTimeout(resolve, delay));
    }
    return false;
}"""

JS_CHECK_TEXT_ELEMENT_STATUS = """(text) => {
    const results = [];
    const all = document.querySelectorAll('*');
    for (const el of all) {
        if (el.offsetParent !== null && el.textContent.includes(text)) {
             let childHasText = false;
             for (const child of el.children) {
                 if (child.textContent.includes(text)) {
                     childHasText = true;
                     break;
                 }
             }
             if (!childHasText) {
                const isChecked = el.checked || false;
                const isSelected = el.selected || false;
                const ariaSelected = el.getAttribute('aria-selected') === 'true';
                const ariaChecked = el.getAttribute('aria-checked') === 'true';
                const ariaPressed = el.getAttribute('aria-pressed') === 'true';
                const classList = el.className || "";
                const hasSelectedClass = /selected|active|checked|toggled|chosen|correct|wrong|answer/i.test(classList);
                
                let parent = el.parentElement;
                let parentSelected = false;
                if (parent) {
                    const pClass = parent.className || "";
                    parentSelected = /selected|active|checked|toggled|chosen|correct|wrong|answer/i.test(pClass) || parent.getAttribute('aria-selected') === 'true';
                }

                results.push({
                    tag: el.tagName.toLowerCase(),
                    text: el.innerText.trim().substring(0, 50),
                    isLikelySelected: isChecked || isSelected || ariaSelected || ariaChecked || ariaPressed || hasSelectedClass || parentSelected,
                    details: { isChecked, isSelected, ariaSelected, hasSelectedClass, parentSelected }
                });
             }
        }
    }
    return results.slice(0, 5);
}"""

JS_CLOSE_COOKIE_BANNERS = """() => {
    // Helper to find elements across Shadow DOMs
    function querySelectorAllDeep(selector, root = document) {
        let elements = Array.from(root.querySelectorAll(selector));
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
        let node;
        while (node = walker.nextNode()) {
            if (node.shadowRoot) {
                elements = elements.concat(querySelectorAllDeep(selector, node.shadowRoot));
            }
        }
        return elements;
    }

    const selectors = [
        '#onetrust-accept-btn-handler',
        '#onetrust-reject-all-handler',
        '.cc-btn.cc-dismiss',
        '.cc-btn.cc-allow',
        '.fc-cta-consent', 
        '.fc-primary-button',
        '[aria-label="Accept cookies"]',
        '[aria-label="Allow cookies"]',
        'button[class*="cookie"][class*="accept"]',
        'button[class*="cookie"][class*="allow"]',
        'button[class*="consent"][class*="accept"]',
        'button[class*="consent"][class*="allow"]',
        'button[id*="cookie"][id*="accept"]',
        '#cmp-welcome-button',
        '.cmp-button',
        'button.sc-ifAKCX.ljEJIv',
        'a.cc-btn.cc-accept-all',
        'button[data-testid="uc-accept-all-button"]',
        'button.osano-cm-accept-all',
        'button.cl-accept-all'
    ];

    const keywords = ['accept', 'agree', 'allow', 'consent', 'continue', 'ok', 'i understand', 'got it', 'accept all', 'allow all', 'accept selection'];
    
    const isVisible = (el) => {
        const style = window.getComputedStyle(el);
        return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0' && el.offsetParent !== null;
    };

    // 1. Try specific selectors (Deep search)
    for (const selector of selectors) {
        const elements = querySelectorAllDeep(selector);
        for (const element of elements) {
            if (isVisible(element)) {
                element.click();
                return true;
            }
        }
    }

    // 2. Scan buttons for keywords (Deep search)
    const candidates = querySelectorAllDeep('button, a, div[role="button"], input[type="button"], input[type="submit"]');
    
    for (const el of candidates) {
        const text = (el.innerText || el.value || '').toLowerCase().trim();
        if (text.length < 2 || text.length > 50) continue;

        const isStrongMatch = ['accept', 'agree', 'allow all', 'accept all cookies', 'i agree', 'got it', 'ok'].includes(text);
        const matchesKeyword = keywords.some(kw => text.includes(kw));
        
        if (matchesKeyword && isVisible(el)) {
            // Check if it looks like a banner (fixed/sticky parent)
            let parent = el.parentElement;
            let isBanner = false;
            let depth = 0;
            while (parent && depth < 5) {
                if (parent instanceof ShadowRoot) {
                    parent = parent.host;
                    continue;
                }
                const pStyle = window.getComputedStyle(parent);
                if (pStyle.position === 'fixed' || pStyle.position === 'sticky' || parseInt(pStyle.zIndex) > 100 || pStyle.bottom === '0px') {
                    isBanner = true;
                    break;
                }
                if (parent.getAttribute('role') === 'dialog' || parent.getAttribute('role') === 'alertdialog') {
                    isBanner = true;
                    break;
                }
                parent = parent.parentElement;
                depth++;
            }

            if (isBanner || isStrongMatch) {
                el.click();
                return true;
            }
        }
    }

    return false;
}"""

JS_DETECT_BLOCKING_ELEMENTS = """() => {
    const elements = document.querySelectorAll('div, section, iframe, dialog, .popup, .modal, [role="dialog"]');
    for (const el of elements) {
        const style = window.getComputedStyle(el);
        if ((style.position === 'fixed' || style.position === 'absolute') && parseInt(style.zIndex) > 50) {
            const rect = el.getBoundingClientRect();
            if (rect.width > window.innerWidth * 0.6 && rect.height > window.innerHeight * 0.6) {
                 if (style.opacity > 0.1 && style.visibility !== 'hidden' && style.display !== 'none') {
                     return true;
                 }
            }
        }
    }
    return false;
}"""

JS_HANDLE_VIGNETTE = """() => {
    const vignette = document.getElementById('google_vignette');
    if (vignette) {
        vignette.remove();
        // Google Vignettes often lock scrolling on the body/html
        document.body.style.overflow = 'auto';
        document.documentElement.style.overflow = 'auto';
        return true;
    }
    return false;
}"""

JS_EXTRACT_TABLES = """() => {
    return Array.from(document.querySelectorAll('table')).map((table, index) => {
        if (table.offsetParent === null) return null;
        
        const caption = table.caption ? table.caption.innerText.trim() : '';
        const rows = Array.from(table.querySelectorAll('tr')).map(tr => {
            return Array.from(tr.querySelectorAll('td, th')).map(cell => cell.innerText.replace(/\\s+/g, ' ').trim());
        }).filter(row => row.length > 0);
        
        return { index, caption, rows };
    }).filter(t => t && t.rows.length > 0);
}"""

JS_SMART_SCROLL = """(targetText) => {
    const findTarget = (text) => {
        const query = text.toLowerCase();
        // Priority search: headers, bold text, labels, then paragraphs/spans
        const selectors = ['h1', 'h2', 'h3', 'b', 'strong', 'label', 'span', 'p', 'div', 'button', 'a'];
        for (const selector of selectors) {
            const elements = Array.from(document.querySelectorAll(selector));
            const found = elements.find(el => {
                const innerText = (el.innerText || "").toLowerCase();
                return innerText.includes(query) && el.offsetParent !== null; // Ensure element is visible
            });
            if (found) return found;
        }
        return null;
    };

    const target = findTarget(targetText);
    
    if (target) {
        // Find a semantic container (e.g. a card, a section, or a bordered div)
        let container = target.parentElement;
        let depth = 0;
        while (container && container.tagName !== 'BODY' && depth < 5) {
            const style = window.getComputedStyle(container);
            const hasBorder = style.borderWidth !== '0px';
            const isQuizClass = /quiz|question|task|assignment/i.test(container.className + container.id);
            
            if (hasBorder || isQuizClass) {
                break;
            }
            container = container.parentElement;
            depth++;
        }

        const finalTarget = (container && container.tagName !== 'BODY') ? container : target;
        
        finalTarget.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Highlight logic
        const overlayId = 'ai-focus-overlay';
        let overlay = document.getElementById(overlayId);
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = overlayId;
            document.body.appendChild(overlay);
        }
        
        const rect = finalTarget.getBoundingClientRect();
        setTimeout(() => {
            Object.assign(overlay.style, {
                position: 'fixed',
                top: rect.top + 'px',
                left: rect.left + 'px',
                width: rect.width + 'px',
                height: rect.height + 'px',
                border: '4px solid #00FF00',
                pointerEvents: 'none',
                zIndex: '2147483647',
                boxShadow: '0 0 0 9999px rgba(0,0,0,0.6)',
                borderRadius: '8px',
                transition: 'all 0.3s ease-out'
            });
        }, 300);
        
        return true;
    }
    return false;
}"""

JS_PURIFY_DOM = """() => {
    const adSelectors = [
        '#google_vignette', 
        '.adsbygoogle', 
        'iframe[id*="google_ads"]',
        '.vignette-container',
        'ins.adsbygoogle',
        '.fc-ab-root',
        '[id^="ad_unit"]',
        '.premium-ad'
    ];
    
    const clean = () => {
        adSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => el.remove());
        });
        document.body.style.overflow = 'auto';
        document.documentElement.style.overflow = 'auto';
        const allElements = document.querySelectorAll('div');
        allElements.forEach(el => {
            const zIndex = parseInt(window.getComputedStyle(el).zIndex);
            if (zIndex > 1000 && !el.id.includes('ai-focus-overlay')) {
                const rect = el.getBoundingClientRect();
                if (rect.width > window.innerWidth * 0.5 && rect.height > window.innerHeight * 0.5) {
                    el.remove();
                }
            }
        });
    };
    clean();
    setTimeout(clean, 2000);
    return true;
}"""

JS_EXTRACT_LIST_ITEMS = """() => {
    const lists = Array.from(document.querySelectorAll('ul, ol, div[role="list"]'));
    return lists.map(list => {
        const items = Array.from(list.querySelectorAll('li, div[role="listitem"]'))
            .map(li => li.innerText.trim())
            .filter(t => t.length > 0);
        
        if (items.length < 2) return null;
        
        return {
            type: list.tagName.toLowerCase(),
            items: items.slice(0, 50),
            count: items.length
        };
    }).filter(l => l !== null);
}"""

JS_GET_MAIN_CONTENT = """() => {
    // Heuristic to find the main content area
    const candidates = ['article', 'main', '[role="main"]', '#content', '.content', '#main', '.main', '.post', '.article'];
    for (const selector of candidates) {
        const el = document.querySelector(selector);
        // Ensure it has substantial content
        if (el && el.innerText.trim().length > 200) {
            return el.innerText.trim();
        }
    }
    // Fallback: Return body text if no clear main container found
    return document.body.innerText.trim();
}"""

JS_EXTRACT_IMAGES = """() => {
    const images = Array.from(document.querySelectorAll('img'));
    return images.map(img => {
        const rect = img.getBoundingClientRect();
        const isVisible = rect.width > 10 && rect.height > 10 && window.getComputedStyle(img).visibility !== 'hidden';
        if (!isVisible) return null;
        return {
            src: img.src,
            alt: img.alt || '',
            width: Math.round(rect.width),
            height: Math.round(rect.height)
        };
    }).filter(img => img !== null && img.src.startsWith('http')).slice(0, 20);
}"""

JS_VERIFY_ELEMENT_STATE = """([selector, state]) => {
    const el = document.querySelector(selector);
    if (!el) return { found: false };
    
    const style = window.getComputedStyle(el);
    const isVisible = style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0' && el.offsetParent !== null;
    
    let result = false;
    switch (state.toLowerCase()) {
        case 'checked': result = el.checked || el.getAttribute('aria-checked') === 'true'; break;
        case 'unchecked': result = !el.checked && el.getAttribute('aria-checked') !== 'true'; break;
        case 'visible': result = isVisible; break;
        case 'hidden': result = !isVisible; break;
        case 'enabled': result = !el.disabled && el.getAttribute('aria-disabled') !== 'true'; break;
        case 'disabled': result = el.disabled || el.getAttribute('aria-disabled') === 'true'; break;
        case 'editable': result = !el.readOnly && !el.disabled; break;
        case 'readonly': result = el.readOnly; break;
        case 'selected': result = el.selected || el.getAttribute('aria-selected') === 'true'; break;
    }
    return { found: true, match: result, tagName: el.tagName };
}"""

JS_SET_DATE_VALUE = """(el, value) => {
    try {
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        el.dispatchEvent(new Event('blur', { bubbles: true }));
        return { success: true };
    } catch (e) {
        return { success: false, error: e.toString() };
    }
}"""

JS_SCROLL_TO_SELECTOR = """async (selector) => {
    const maxScrolls = 60;
    const distance = 600;
    const delay = 100;
    
    for (let i = 0; i < maxScrolls; i++) {
        const element = document.querySelector(selector);
        if (element) {
            const rect = element.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
                element.scrollIntoView({behavior: 'smooth', block: 'center'});
                return true;
            }
        }
        if ((window.innerHeight + window.scrollY) >= document.documentElement.scrollHeight - 10) {
            break;
        }
        window.scrollBy(0, distance);
        await new Promise(resolve => setTimeout(resolve, delay));
    }
    return false;
}"""

JS_SET_RANGE_VALUE = """(el, value) => {
    try {
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        return { success: true };
    } catch (e) {
        return { success: false, error: e.toString() };
    }
}"""

JS_GET_ATTRIBUTES = """([selector, attrNames]) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    
    const result = {};
    const names = attrNames ? attrNames.split(',').map(s => s.trim()).filter(s => s.length > 0) : [];
    
    if (names.length > 0) {
        names.forEach(name => {
            result[name] = el.getAttribute(name);
        });
    } else {
        for (const attr of el.attributes) {
            result[attr.name] = attr.value;
        }
    }
    return result;
}"""

JS_SET_COLOR_VALUE = """(el, value) => {
    try {
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
        return { success: true };
    } catch (e) {
        return { success: false, error: e.toString() };
    }
}"""

JS_CLOSE_NEWSLETTER = """() => {
    const keywords = ['no thanks', 'maybe later', 'continue to site', 'not now', 'close', 'later'];
    const containers = document.querySelectorAll('div[class*="newsletter"], div[class*="subscribe"], div[class*="modal"], div[role="dialog"], div[class*="popup"]');
    
    for (const container of containers) {
        if (container.offsetParent === null) continue; // Invisible
        
        // Look for close button or "No thanks" link inside
        const buttons = container.querySelectorAll('button, a, svg, [role="button"]');
        for (const btn of buttons) {
            const text = (btn.innerText || '').toLowerCase();
            const label = (btn.getAttribute('aria-label') || '').toLowerCase();
            if (keywords.some(kw => text.includes(kw)) || label.includes('close')) {
                btn.click();
                return true;
            }
        }
    }
    return false;
}"""

JS_GET_ELEMENT_COORDINATES = """(selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        top: rect.top,
        left: rect.left,
        bottom: rect.bottom,
        right: rect.right,
        centerX: rect.left + rect.width / 2,
        centerY: rect.top + rect.height / 2,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight
    };
}"""

JS_HANDLE_RATE_EXPERIENCE = """() => {
    const keywords = ['rate this experience', 'feedback', 'how likely are you', 'survey', 'opinion', 'satisfied', 'recommend us'];
    const containers = document.querySelectorAll('div, section, aside, [role="dialog"]');
    
    for (const container of containers) {
        if (container.offsetParent === null) continue;
        const style = window.getComputedStyle(container);
        if (!['fixed', 'absolute', 'sticky'].includes(style.position) && container.tagName !== 'DIALOG') continue;
        const text = container.innerText.toLowerCase();
        if (text.length > 500) continue;
        if (keywords.some(kw => text.includes(kw))) {
            const closeBtn = container.querySelector('button[aria-label*="close"], button[class*="close"], svg[data-icon="close"], .close, [aria-label="Dismiss"]');
            if (closeBtn) { closeBtn.click(); return true; }
            const buttons = container.querySelectorAll('button, a');
            for (const btn of buttons) {
                if (['no thanks', 'not now', 'later', 'dismiss', 'close', 'maybe later'].some(s => btn.innerText.toLowerCase().includes(s))) {
                    btn.click(); return true;
                }
            }
        }
    }
    return false;
}"""

JS_GET_PAGE_METADATA = """() => {
    const getMeta = (name) => {
        const el = document.querySelector(`meta[name="${name}"], meta[property="${name}"]`);
        return el ? el.getAttribute('content') : null;
    };
    return {
        title: document.title,
        description: getMeta('description') || getMeta('og:description') || getMeta('twitter:description'),
        keywords: getMeta('keywords'),
        ogTitle: getMeta('og:title'),
        ogImage: getMeta('og:image'),
        ogUrl: getMeta('og:url'),
        canonical: document.querySelector('link[rel="canonical"]')?.href
    };
}"""

JS_HANDLE_NOTIFICATIONS_PROMPT = """() => {
    const keywords = ['allow notifications', 'enable notifications', 'show notifications', 'receive updates', 'push notifications'];
    const containers = document.querySelectorAll('div, section, [role="dialog"]');
    
    for (const container of containers) {
        if (container.offsetParent === null) continue;
        const text = container.innerText.toLowerCase();
        if (text.length > 300) continue;
        
        if (keywords.some(kw => text.includes(kw))) {
            // Look for deny/close buttons
            const buttons = container.querySelectorAll('button, a');
            for (const btn of buttons) {
                const btnText = btn.innerText.toLowerCase();
                if (['block', 'deny', 'no thanks', 'later', 'not now', 'close', 'dismiss'].some(s => btnText.includes(s))) {
                    btn.click();
                    return true;
                }
            }
            // Fallback: try to find a close icon
            const closeIcon = container.querySelector('button[aria-label*="close"], .close');
            if (closeIcon) { closeIcon.click(); return true; }
        }
    }
    return false;
}"""

JS_HANDLE_INSTALL_APP = """() => {
    const keywords = ['install app', 'get the app', 'open in app', 'download app', 'continue in app'];
    const containers = document.querySelectorAll('div, section, [role="banner"], [role="dialog"]');
    
    for (const container of containers) {
        if (container.offsetParent === null) continue;
        const text = container.innerText.toLowerCase();
        if (text.length > 200) continue; // Banners are usually small
        
        if (keywords.some(kw => text.includes(kw))) {
            // Look for close/dismiss/not now
            const buttons = container.querySelectorAll('button, a, svg, [role="button"]');
            for (const btn of buttons) {
                const btnText = (btn.innerText || btn.getAttribute('aria-label') || '').toLowerCase();
                if (['close', 'dismiss', 'not now', 'continue to website', 'expand', 'x'].some(s => btnText.includes(s))) {
                    btn.click();
                    return true;
                }
            }
        }
    }
    return false;
}"""

JS_HANDLE_AGE_GATE = """() => {
    const keywords = ['enter', 'confirm', 'i am over 18', 'i am 21 or older', 'yes, i am of legal age', 'submit'];
    const containers = document.querySelectorAll('div, section, [role="dialog"]');
    
    for (const container of containers) {
        if (container.offsetParent === null) continue;
        const text = container.innerText.toLowerCase();
        if (text.length > 400) continue;
        
        if (text.includes('age verification') || text.includes('are you 18') || text.includes('are you 21')) {
            const buttons = container.querySelectorAll('button, a, input[type="submit"]');
            for (const btn of buttons) {
                const btnText = (btn.innerText || btn.value || '').toLowerCase();
                if (keywords.some(kw => btnText.includes(kw))) {
                    btn.click();
                    return true;
                }
            }
        }
    }
    return false;
}"""

JS_DETECT_NAVIGATION_CONTROLS = """() => {
    const text = document.body.innerText;
    // Detect wizard steps (e.g., "Step 1 of 3", "Question 5/10")
    const stepMatch = text.match(/(?:Step|Stage|Question)\\s*\\d+\\s*(?:of|\\/)\\s*\\d+/i);
    
    // Detect Logged In State (Priority)
    const loggedInMatch = text.match(/(?:Log out|Sign out|My Account|Profile|Dashboard|Courses|Classes|Welcome|Student|Canvas|Blackboard)/i);

    // Detect Auth/Login
    const authMatch = text.match(/(?:Sign in|Log in|Welcome back|Forgot password)/i);
    const authInputs = document.querySelectorAll('input[type="password"]');
    
    // Detect Newsletter/Popups
    const newsletterMatch = text.match(/(?:Subscribe|Newsletter|Join our list|Get updates)/i);
    
    // Detect Rating/Feedback
    const ratingMatch = text.match(/(?:Rate this experience|How likely are you|Feedback|Survey)/i);
    
    // Detect Notification Prompts
    const notifyMatch = text.match(/(?:Allow|Enable|Show) notifications/i);
    
    // Detect App Banners
    const appMatch = text.match(/(?:Install|Get|Open in) app/i);

    // Detect Age Gates
    const ageGateMatch = text.match(/(?:age verification|are you 18|are you 21|enter site)/i);
    
    const nextBtns = Array.from(document.querySelectorAll('a, button, input[type="submit"]')).filter(el => 
        (el.innerText && /next|continue|proceed|forward|>/i.test(el.innerText)) || 
        (el.getAttribute('aria-label') && /next|continue|forward/i.test(el.getAttribute('aria-label')))
    );
    const loadMoreBtns = Array.from(document.querySelectorAll('button, a')).filter(el =>
        (el.innerText && /load more|show more|view more/i.test(el.innerText))
    );
    
    let msg = "";
    if (loggedInMatch) {
        msg += "Status: User appears to be Logged In. Proceed with task. ";
    } else if (authMatch || authInputs.length > 0) {
        msg += "Auth Detected: Login form or text found. ";
    }
    
    if (ageGateMatch) msg += "Age Gate Detected: You may need to 'dismiss_age_gate'. ";
    if (newsletterMatch) msg += "Newsletter Detected: You may need to 'close_newsletter_modal'. ";
    if (ratingMatch) msg += "Rating Popup Detected: You may need to 'dismiss_rating_popup'. ";
    if (notifyMatch) msg += "Notification Prompt Detected: You may need to 'dismiss_notification_prompt'. ";
    if (appMatch) msg += "App Banner Detected: You may need to 'dismiss_app_banner' or 'clear_view'. ";
    if (stepMatch) msg += `Wizard Progress: ${stepMatch[0]}. `;
    if (nextBtns.length > 0) msg += "Pagination: 'Next' button available. ";
    if (loadMoreBtns.length > 0) msg += "Pagination: 'Load More' button available. ";
    return msg.trim();
}"""

JS_GET_CONSOLE_ERRORS = """() => {
    return window._captured_logs ? window._captured_logs.filter(l => l.type === 'error') : [];
}"""

JS_INJECT_HUD = """(data) => {
    // data: { plan: [], goal: string, status: string, last_action: string }
    if (!document.body) return;
    
    const id = 'agent-hud-bottom-panel';
    let container = document.getElementById(id);

    // Initialize state
    if (!window._agent_hud_state) {
        window._agent_hud_state = { collapsed: false };
    }

    // Toggle function
    if (!window._agent_hud_toggle) {
        window._agent_hud_toggle = () => {
            window._agent_hud_state.collapsed = !window._agent_hud_state.collapsed;
            const c = document.getElementById('agent-hud-bottom-panel');
            const content = document.getElementById('agent-hud-bottom-panel-content');
            const btn = document.getElementById('agent-hud-toggle-btn');
            
            if (window._agent_hud_state.collapsed) {
                if(content) content.style.display = 'none';
                if(c) c.style.height = '40px';
                if(btn) btn.innerText = '🔼';
                document.body.style.marginBottom = '40px';
            } else {
                if(content) content.style.display = 'flex';
                if(c) c.style.height = '300px';
                if(btn) btn.innerText = '🔽';
                document.body.style.marginBottom = '300px';
            }
        };
    }

    // Re-create if needed
    if (container && !document.getElementById(id + '-input-row')) {
        container.remove();
        container = null;
    }

    if (!container) {
        container = document.createElement('div');
        container.id = id;
        Object.assign(container.style, {
            position: 'fixed',
            bottom: '0',
            left: '0',
            width: '100%',
            backgroundColor: '#0f172a',
            color: '#e2e8f0',
            zIndex: '2147483647',
            fontFamily: 'Segoe UI, sans-serif',
            boxSizing: 'border-box',
            borderTop: '2px solid #3b82f6',
            boxShadow: '0 -4px 20px rgba(0,0,0,0.4)',
            display: 'flex',
            flexDirection: 'column',
            height: '300px',
            transition: 'height 0.3s ease',
        });
        document.body.appendChild(container);

        // Header
        const header = document.createElement('div');
        header.id = id + '-header';
        Object.assign(header.style, {
            padding: '0 16px',
            height: '40px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'pointer',
            borderBottom: '1px solid #1e293b',
            backgroundColor: '#1e293b',
            flexShrink: '0'
        });
        header.onclick = (e) => {
            if (e.target === header || e.target.parentElement === header) {
                 window._agent_hud_toggle();
            }
        };
        container.appendChild(header);

        // Content Container
        const content = document.createElement('div');
        content.id = id + '-content';
        Object.assign(content.style, {
            padding: '12px 16px',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            flex: '1',
            backgroundColor: '#0f172a'
        });
        container.appendChild(content);
        
        // Input Row
        const inputRow = document.createElement('div');
        inputRow.id = id + '-input-row';
        Object.assign(inputRow.style, {
            display: 'flex',
            gap: '8px',
            marginBottom: '8px'
        });
        
        const input = document.createElement('input');
        input.id = id + '-input';
        input.placeholder = "Type instruction to agent...";
        Object.assign(input.style, {
            flex: '1',
            padding: '8px 12px',
            borderRadius: '6px',
            border: '1px solid #334155',
            background: '#1e293b',
            color: '#e2e8f0',
            fontSize: '13px',
            outline: 'none'
        });
        
        const addBtn = document.createElement('button');
        addBtn.innerText = "Send";
        Object.assign(addBtn.style, {
            padding: '8px 16px',
            borderRadius: '6px',
            border: 'none',
            background: '#3b82f6',
            color: 'white',
            cursor: 'pointer',
            fontSize: '13px',
            fontWeight: '600'
        });
        
        addBtn.onclick = () => {
            const val = input.value.trim();
            if(val && window.py_agent_control) {
                window.py_agent_control('add_task', val);
                input.value = "";
            }
        };
        input.onkeydown = (e) => { if(e.key === 'Enter') addBtn.click(); };
        
        inputRow.appendChild(input);
        inputRow.appendChild(addBtn);
        content.appendChild(inputRow);

        // Columns Wrapper
        const columnsWrapper = document.createElement('div');
        Object.assign(columnsWrapper.style, {
            display: 'flex',
            gap: '20px',
            flex: '1',
            minHeight: '0' // Important for nested scrolling
        });
        content.appendChild(columnsWrapper);

        // Plan Column
        const planColumn = document.createElement('div');
        planColumn.id = id + '-plan-column';
        Object.assign(planColumn.style, {
            flex: '1',
            overflowY: 'auto',
            paddingRight: '8px'
        });
        columnsWrapper.appendChild(planColumn);
        
        // Status Column
        const statusColumn = document.createElement('div');
        statusColumn.id = id + '-status-column';
        Object.assign(statusColumn.style, {
            flex: '1',
            overflowY: 'auto',
            paddingRight: '8px',
            borderLeft: '1px solid #334155',
            paddingLeft: '20px'
        });
        columnsWrapper.appendChild(statusColumn);
    }
    
    // Update Content
    const header = document.getElementById(id + '-header');
    const content = document.getElementById(id + '-content');
    const planColumn = document.getElementById(id + '-plan-column');
    const statusColumn = document.getElementById(id + '-status-column');

    if (!header || !content || !planColumn || !statusColumn) return;

    // Apply state
    if (window._agent_hud_state.collapsed) {
        content.style.display = 'none';
        container.style.height = '40px';
        document.body.style.marginBottom = '40px';
    } else {
        content.style.display = 'flex';
        container.style.height = '300px';
        document.body.style.marginBottom = '300px';
    }

    // Update Header
    const goalText = data.goal ? data.goal.substring(0, 100) : 'Ready';
    const btnStyle = "margin-left:8px; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px; font-weight:600; border:none; transition: opacity 0.2s;";
    const toggleIcon = window._agent_hud_state.collapsed ? '🔼' : '🔽';

    header.innerHTML = `
        <div style="display:flex; align-items:center; gap:12px; flex:1; overflow:hidden;">
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="font-size:16px;">🤖</span>
                <span style="font-weight:700; color:#38bdf8; font-size:13px; letter-spacing:0.5px;">AI BROWSER</span>
            </div>
            <div style="height:16px; width:1px; background:#334155;"></div>
            <span style="font-size:12px; color:#94a3b8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${goalText}</span>
        </div>
        <div style="display:flex; align-items:center;">
            <button onclick="if(window.py_agent_control) window.py_agent_control('pause')" style="${btnStyle} background:#334155; color:#e2e8f0;">⏸️ Pause</button>
            <button onclick="if(window.py_agent_control) window.py_agent_control('stop')" style="${btnStyle} background:#ef4444; color:white;">⏹️ Stop</button>
            <button id="agent-hud-toggle-btn" onclick="window._agent_hud_toggle()" style="${btnStyle} background:transparent; color:#94a3b8; font-size:14px;">${toggleIcon}</button>
        </div>
    `;

    // Update Plan
    planColumn.innerHTML = `<div style="font-size:11px; font-weight:700; color:#64748b; margin-bottom:8px; text-transform:uppercase;">Current Plan</div>`;
    if (data.plan && data.plan.length > 0) {
        data.plan.forEach((step, idx) => {
            // Show pending, in_progress, failed, and the last completed step
            let lastCompletedIdx = -1;
            data.plan.forEach((s, i) => { if(s.status === 'completed') lastCompletedIdx = i; });
            
            const isLastCompleted = idx === lastCompletedIdx;
            // Show all steps for better context, but style completed ones differently
            const shouldShow = true; 
            
            if (shouldShow) {
                const item = document.createElement('div');
                let border = '#334155';
                let bg = 'transparent';
                let icon = '○';
                let color = '#94a3b8';
                
                if (step.status === 'in_progress') {
                    border = '#3b82f6';
                    bg = 'rgba(59, 130, 246, 0.1)';
                    icon = '▶';
                    color = '#e2e8f0';
                } else if (step.status === 'completed') {
                    border = '#22c55e';
                    icon = '✓';
                    color = '#22c55e';
                    // Optional: dim completed steps
                } else if (step.status === 'failed') {
                    border = '#ef4444';
                    icon = '✕';
                    color = '#ef4444';
                }

                Object.assign(item.style, {
                    padding: '8px',
                    marginBottom: '6px',
                    borderRadius: '4px',
                    borderLeft: `3px solid ${border}`,
                    backgroundColor: bg,
                    fontSize: '12px',
                    color: color,
                    display: 'flex',
                    gap: '8px'
                });
                item.innerHTML = `<span>${icon}</span><span>${step.step}</span>`;
                planColumn.appendChild(item);
            }
        });
    } else {
        planColumn.innerHTML += `<div style="color:#475569; font-size:12px; font-style:italic;">No active plan.</div>`;
    }

    // Update Status/Logs
    statusColumn.innerHTML = `<div style="font-size:11px; font-weight:700; color:#64748b; margin-bottom:8px; text-transform:uppercase;">Last Action</div>`;
    if (data.last_action) {
        const log = document.createElement('div');
        Object.assign(log.style, {
            fontFamily: 'monospace',
            fontSize: '11px',
            color: '#a5b4fc',
            backgroundColor: 'rgba(0,0,0,0.2)',
            padding: '8px',
            borderRadius: '4px',
            lineHeight: '1.4'
        });
        log.innerText = data.last_action;
        statusColumn.appendChild(log);
    }
}"""

JS_MONITOR_MUTATIONS = """(timeout) => {
    return new Promise((resolve) => {
        let activeContainer = null;
        let maxChanges = 0;
        const mutationCounts = new Map();
        
        const observer = new MutationObserver((mutations) => {
            for (const m of mutations) {
                let target = m.target;
                if (target.nodeType === 3) target = target.parentElement;
                
                let curr = target;
                let depth = 0;
                while (curr && curr.tagName !== 'BODY' && depth < 5) {
                    if (['DIV', 'ARTICLE', 'SECTION', 'MAIN', 'UL', 'OL', 'TBODY'].includes(curr.tagName)) {
                        const count = (mutationCounts.get(curr) || 0) + 1;
                        mutationCounts.set(curr, count);
                        if (count > maxChanges) {
                            maxChanges = count;
                            activeContainer = curr;
                        }
                        break; 
                    }
                    curr = curr.parentElement;
                    depth++;
                }
            }
        });
        
        observer.observe(document.body, { childList: true, subtree: true, characterData: true });
        
        setTimeout(() => {
            observer.disconnect();
            if (activeContainer && maxChanges > 1) { // Threshold for noise
                let selector = activeContainer.tagName.toLowerCase();
                if (activeContainer.id) selector += '#' + activeContainer.id;
                else if (activeContainer.className && typeof activeContainer.className === 'string') {
                     const classes = activeContainer.className.trim().split(/\\s+/).filter(c => c.length > 0);
                     if (classes.length > 0) selector += '.' + classes.join('.');
                }
                
                resolve({
                    detected: true,
                    selector: selector,
                    change_count: maxChanges,
                    new_content_preview: activeContainer.innerText ? activeContainer.innerText.substring(0, 500) : "No text"
                });
            } else {
                resolve({ detected: false });
            }
        }, timeout);
    });
}"""

JS_IDENTIFY_MAIN_CONTAINER = """() => {
    const candidates = ['article', 'main', '[role="main"]', '#content', '.content', '#main', '.main', '.feed', '[role="feed"]', '.timeline'];
    for (const selector of candidates) {
        const el = document.querySelector(selector);
        if (el && el.offsetHeight > 200 && el.innerText.length > 200) {
            let path = el.tagName.toLowerCase();
            if (el.id) path += '#' + el.id;
            else if (el.className && typeof el.className === 'string') {
                const classes = el.className.trim().split(/\\s+/).filter(c => c.length > 0);
                if (classes.length > 0) path += '.' + classes.join('.');
            }
            return { selector: path, text_preview: el.innerText.substring(0, 200) };
        }
    }
    return { selector: 'body', text_preview: document.body.innerText.substring(0, 200) };
}"""