from collections import namedtuple

import numpy as np

from .vector import Vector


class Electromagnetism:
    """Electrostatic and magnetostatic calculations using point-charge models."""

    K_E: float = 8.9875517923e9
    MU_0: float = 1.25663706212e-6
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

    @staticmethod
    def electric_potential(charges, point) -> float:
        """Calculate the net electric potential at a point from point charges.

        Args:
            charges: Iterable of Charge namedtuples, each with .q and .position.
            point: Coordinates of the observation point.

        Returns:
            float: Net electric potential (volts).
        """
        V = 0.0
        for charge in charges:
            r_vec = np.array(point) - np.array(charge.position)
            r = np.sqrt(np.sum(r_vec ** 2))
            if r < 1e-18:
                continue
            V += Electromagnetism.K_E * charge.q / r
        return V

    @staticmethod
    def calculate_magnetic_field(charges, point, velocities) -> Vector:
        """Calculate the magnetic field at a point from moving point charges using Biot-Savart.

        Args:
            charges: Iterable of Charge namedtuples, each with .q and .position.
            point: Coordinates of the observation point.
            velocities: Iterable of velocity 3-tuples/vectors, one per charge.

        Returns:
            Vector: Net magnetic field vector at the point (tesla).
        """
        B_total = np.zeros(3)
        for charge, v in zip(charges, velocities):
            r_vec = np.array(point) - np.array(charge.position)
            r_mag = np.sqrt(np.sum(r_vec ** 2))
            if r_mag < 1e-18:
                continue
            v_arr = np.array(v)
            B_total += (Electromagnetism.MU_0 / (4.0 * np.pi)) * charge.q * np.cross(v_arr, r_vec) / r_mag ** 3
        return Vector(B_total.tolist())

    @staticmethod
    def dipole_moment(charges) -> Vector:
        """Calculate the net electric dipole moment of a charge distribution.

        Args:
            charges: Iterable of Charge namedtuples, each with .q and .position.

        Returns:
            Vector: Net dipole moment vector (C·m).
        """
        p = np.zeros(3)
        for charge in charges:
            p += charge.q * np.array(charge.position)
        return Vector(p.tolist())

    @staticmethod
    def combine_fields(fields) -> Vector:
        """Superpose multiple field vectors into one net field.

        Args:
            fields: Iterable of Vector or array-like field vectors.

        Returns:
            Vector: Sum of all input fields.
        """
        total = np.zeros_like(np.array(fields[0]))
        for f in fields:
            total += np.array(f)
        return Vector(total.tolist())
