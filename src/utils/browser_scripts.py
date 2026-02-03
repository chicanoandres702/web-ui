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
    return anchors.map(a => ({
        text: a.innerText.trim() || a.getAttribute('aria-label') || '',
        href: a.href
    })).filter(link => link.href.startsWith('http') && link.text.length > 0);
}"""

JS_GET_STRUCTURE = """() => {
    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6'));
    return headings.map(h => ({
        tag: h.tagName,
        text: h.innerText.trim()
    })).filter(h => h.text.length > 0);
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
        '.preroll-ad'
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
    const commonSelectors = [
        '#onetrust-accept-btn-handler',
        '#onetrust-reject-all-handler',
        '.cc-btn.cc-dismiss',
        '.cc-btn.cc-allow',
        'button[class*="cookie"][class*="accept"]',
        'button[class*="cookie"][class*="allow"]',
        'button[class*="consent"][class*="accept"]',
        'button[class*="consent"][class*="allow"]',
        'button[id*="cookie"][id*="accept"]',
        'button[id*="cookie"][id*="allow"]'
    ];
    
    let clicked = false;
    for (const selector of commonSelectors) {
        const element = document.querySelector(selector);
        if (element) {
            element.click();
            clicked = true;
        }
    }
    return clicked;
}"""