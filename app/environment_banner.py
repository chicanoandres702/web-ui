def get_environment_banner_html(args) -> str:

    if args.https:
        return ""

    banner_text = "Warning: Running in HTTP mode. HTTPS is recommended for security."
    banner_style = """
    <div style="background-color: #FFCCCC; color: #000000; padding: 10px; text-align: center; font-weight: bold;">
        {banner_text}
    </div>
    """
    banner_html = banner_style.format(banner_text=banner_text)

    return banner_html