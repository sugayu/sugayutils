'''This program gives hacks to use MLModern in matplotlib.
'''

import unicodedata
import matplotlib
from matplotlib import cbook
import matplotlib.pyplot as plt
from matplotlib.ft2font import FT2Font
from matplotlib._mathtext import (
    _log,
    get_unicode_index,
    get_font,
    StixFonts,
    BakomaFonts,
)


def main():
    # Add tex commands
    tex2uni = matplotlib._mathtext_data.tex2uni
    tex2uni['micro'] = 0x00B5
    matplotlib._mathtext_data.tex2uni = tex2uni

    def next_fontname(fontname: str) -> str:
        if fontname == 'it':
            return 'rm'
        if fontname == 'rm':
            return 'cal'
        return ''

    def _get_glyph(
        self, fontname: str, font_class: str, sym: str
    ) -> tuple[FT2Font, int, bool]:
        '''This is based on matplotlib version 3.8.4.'''
        try:
            uniindex = get_unicode_index(sym)
            found_symbol = True
        except ValueError:
            uniindex = ord('?')
            found_symbol = False
            _log.warning("No TeX to Unicode mapping for %a.", sym)

        fontname, uniindex = self._map_virtual_font(fontname, font_class, uniindex)

        new_fontname = fontname

        # Only characters in the "Letter" class should be italicized in 'it'
        # mode.  Greek capital letters should be Roman.
        if found_symbol:
            if fontname == 'it' and uniindex < 0x10000:
                char = chr(uniindex)
                if unicodedata.category(char)[0] != "L" or unicodedata.name(
                    char
                ).startswith("GREEK CAPITAL"):
                    new_fontname = 'rm'

            found_symbol = False
            while (not found_symbol) and new_fontname:
                font = self._get_font(new_fontname)
                if font is not None:
                    if (
                        uniindex in self._cmr10_substitutions
                        and font.family_name == "cmr10"
                    ):
                        font = get_font(cbook._get_data_path("fonts/ttf/cmsy10.ttf"))
                        uniindex = self._cmr10_substitutions[uniindex]
                    glyphindex = font.get_char_index(uniindex)
                    if glyphindex != 0:
                        found_symbol = True
                new_fontname = next_fontname(new_fontname)
            slanted = (new_fontname == 'it') or sym in self._slanted_symbols

        if not found_symbol:
            if self._fallback_font:
                if fontname in ('it', 'regular') and isinstance(
                    self._fallback_font, StixFonts
                ):
                    fontname = 'rm'

                g = self._fallback_font._get_glyph(fontname, font_class, sym)
                family = g[0].family_name
                if family in list(BakomaFonts._fontmap.values()):
                    family = "Computer Modern"
                _log.info("Substituting symbol %s from %s", sym, family)
                return g

            else:
                if fontname in ('it', 'regular') and isinstance(self, StixFonts):
                    return self._get_glyph('rm', font_class, sym)
                _log.warning(
                    "Font %r does not have a glyph for %a [U+%x], "
                    "substituting with a dummy symbol.",
                    new_fontname,
                    sym,
                    uniindex,
                )
                font = self._get_font('rm')
                uniindex = 0xA4  # currency char, for lack of anything better
                slanted = False

        return font, uniindex, slanted

    matplotlib._mathtext.UnicodeFonts._get_glyph = _get_glyph


if plt.rcParams['mathtext.it'].startswith('MLM'):
    main()
