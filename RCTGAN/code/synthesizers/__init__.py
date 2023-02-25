

from ctgan.synthesizers.ctgan import CTGANSynthesizer

__all__ = (
    'CTGANSynthesizer',
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
