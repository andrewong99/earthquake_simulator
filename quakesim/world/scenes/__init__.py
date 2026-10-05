# Earthquake Simulator -- a physically grounded earthquake simulator.
# Copyright (C) 2026 Earthquake Simulator contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Scene registry."""
from . import warehouse, kl_highrise, ranau_house, supermarket, shophouse  # noqa: F401
from ..scene_base import SCENES, Scene, SceneSpec, Viewpoint      # noqa: F401

SCENE_ORDER = ["warehouse", "kl_highrise", "ranau_house", "supermarket", "shophouse"]
