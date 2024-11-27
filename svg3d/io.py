class OBJ:
    @classmethod
    def _parse_line(cls, line):
        """Parse a line into 'words', each containing one or more pieces of data."""

        return [word.split("/") for word in line.split()]
        # return line.split()

    @classmethod
    def _parse_mtl_into_colors(cls, filename: str, keys: tuple[str] = ("Kd",)):
        """Extract diffuse color values (Kd) from an obj .mtl file.

        https://paulbourke.net/dataformats/mtl/
        """
        with open(filename) as f:
            _materials = {}
            key = None

            for line in f:
                words = cls._parse_line(line)
                if not words:
                    continue

                if words[0] == ["newmtl"]:
                    key = words[1]
                    print(key)
                    _materials[key] = {}
                    continue
                elif words[0][0] in keys:
                    _materials[key][words[0]] = words[1:]

            print(_materials)

    # def __init__(self, filename: str):
    #     vertices, faces = [], []

    #     with open(filename) as f:
    #         for line in f:
    #             words = self._parse_line(line)

    #             if not words: continue


if __name__ == "__main__":
    materials = OBJ._parse_mtl_into_colors(filename="1AI0.mtl")
    print("MATS", materials)
