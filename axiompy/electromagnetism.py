from collections import namedtuple

import numpy as np

from .vector import Vector


class Electromagnetism:
    """Electrostatic calculations using point-charge models."""

    K_E: float = 8.9875517923e9
    Charge = namedtuple('Charge', ['q', 'position'])

    @staticmethod
    def calculate_electric_field(charges, point) -> Vector:
        """Calculate the net electric field at a point from a collection of point charges.

        Args:
            charges: Iterable of Charge namedtuples, each with .q and .position.
            point: Coordinates of the observation point.

        Returns:
            Vector: Net electric field vector at the point.
        """
        E_total = np.zeros(len(point))
        for charge in charges:
            r_vec = np.array(point) - np.array(charge.position)
            r_mag_sq = np.sum(r_vec ** 2)
            if r_mag_sq < 1e-18:
                continue
            E_total += (Electromagnetism.K_E * charge.q / r_mag_sq ** 1.5) * r_vec
        return Vector(E_total.tolist())
