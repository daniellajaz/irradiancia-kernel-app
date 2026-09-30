"""Utilidades geoespaciales.

Las columnas del dataset se llaman ``latitude`` y ``longitude`` pero en
realidad contienen coordenadas proyectadas Web Mercator (EPSG:3857) en
metros: ``latitude`` es la X (este) y ``longitude`` es la Y (norte). Se
conservan esos nombres como variables de entrada del modelo y aquí se
convierten a grados WGS84 (EPSG:4326) para el mapa web.

Para clasificar un punto arbitrario del mapa hacen falta sus 7 bandas
espectrales, que solo se conocen en los puntos observados. ``SpectralField``
las estima por interpolación IDW (inverso de la distancia) a partir de los
vecinos más cercanos del mismo satélite.
"""
import numpy as np
from scipy.spatial import cKDTree

R = 6378137.0
BANDS = ["band1", "band2", "band3", "band4", "band5", "band6", "band7"]


def merc_to_lonlat(x, y):
    lon = np.degrees(np.asarray(x, float) / R)
    lat = np.degrees(np.arctan(np.sinh(np.asarray(y, float) / R)))
    return lon, lat


def lonlat_to_merc(lon, lat):
    x = np.radians(np.asarray(lon, float)) * R
    y = R * np.log(np.tan(np.pi / 4 + np.radians(np.asarray(lat, float)) / 2))
    return x, y


class SpectralField:
    """Interpolador IDW de las bandas de un satélite."""

    def __init__(self, df, k=8, power=2.0):
        self.xy = df[["latitude", "longitude"]].to_numpy(float)
        self.bands = df[BANDS].to_numpy(float)
        self.tree = cKDTree(self.xy)
        self.k = min(k, len(df))
        self.power = power

    def features_at(self, x, y):
        """Devuelve la matriz de entrada [x, y, band1..band7] y la distancia
        (m) a la observación más cercana."""
        pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]).astype(float)
        dist, idx = self.tree.query(pts, k=self.k)
        w = 1.0 / np.maximum(dist, 1.0) ** self.power
        exact = dist[:, 0] < 1.0
        w[exact] = 0.0
        w[exact, 0] = 1.0
        w /= w.sum(axis=1, keepdims=True)
        bands = np.einsum("nk,nkb->nb", w, self.bands[idx])
        return np.column_stack([pts, bands]), dist[:, 0]


class StudyGrid:
    """Malla regular (en metros) que cubre la zona de estudio. Solo se
    conservan celdas a menos de ``max_dist`` de alguna observación."""

    def __init__(self, frames, step=2000.0, max_dist=7500.0):
        xy = np.vstack([f[["latitude", "longitude"]].to_numpy(float) for f in frames])
        x0, y0 = xy.min(0) - step
        x1, y1 = xy.max(0) + step
        self.xs = np.arange(x0, x1 + step, step)
        self.ys = np.arange(y1, y0 - step, -step)  # de norte a sur (filas de imagen)
        gx, gy = np.meshgrid(self.xs, self.ys)
        dist, _ = cKDTree(xy).query(np.column_stack([gx.ravel(), gy.ravel()]))
        self.shape = gx.shape
        self.mask = (dist <= max_dist).reshape(self.shape)
        self.x = gx[self.mask]
        self.y = gy[self.mask]
        self.step = step

    def bounds_latlon(self):
        """[[lat_sur, lon_oeste], [lat_norte, lon_este]] para Leaflet."""
        h = self.step / 2
        lon0, lat0 = merc_to_lonlat(self.xs[0] - h, self.ys[-1] - h)
        lon1, lat1 = merc_to_lonlat(self.xs[-1] + h, self.ys[0] + h)
        return [[float(lat0), float(lon0)], [float(lat1), float(lon1)]]

    def to_image(self, values, fill=np.nan):
        img = np.full(self.shape, fill, dtype=float)
        img[self.mask] = values
        return img


# Paleta del notebook de regresión (azul -> verde -> amarillo -> rojo)
PALETTE = [(0, 0, 1), (0, 0.7, 0), (0, 1, 0), (0.9, 1, 0), (1, 0.7, 0), (1, 0.5, 0), (1, 0, 0)]


def irradiance_rgb(values, vmin, vmax):
    """Mapea irradiancia a RGB (0-255) con la paleta anterior."""
    t = np.clip((np.asarray(values, float) - vmin) / (vmax - vmin), 0, 1)
    pos = t * (len(PALETTE) - 1)
    i = np.minimum(pos.astype(int), len(PALETTE) - 2)
    f = (pos - i)[..., None]
    pal = np.asarray(PALETTE, float)
    rgb = pal[i] * (1 - f) + pal[i + 1] * f
    return (rgb * 255).round().astype(np.uint8)


def rgb_hex(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)
