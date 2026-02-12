import logging
import os
import sys

logger = logging.getLogger(__name__)

def create_certificate(cert_path="./certs", common_name="localhost"):
    """
    Creates self-signed certificates for HTTPS using certipy.
    """
    try:
        import certipy
    except ImportError:
        logger.error("HTTPS enabled but 'certipy' is not installed. Please install it or run without --https.")
        sys.exit(1)

    certificate = certipy.CertificateAuthority(
        path=cert_path,
        common_name=common_name,
    )

    certificate.generate_certificate(
        hosts=["localhost"],
    )

    if not os.path.exists(os.path.join(cert_path, "private.key")) or not os.path.exists(os.path.join(cert_path, "certificate.crt")):
        sys.exit(1)