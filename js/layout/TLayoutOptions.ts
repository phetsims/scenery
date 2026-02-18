// Copyright 2021-2026, University of Colorado Boulder

/**
 * The main type interface for Node's layoutOptions (for use with Grid/Flow based layouts)
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import type { FlowCellOptions } from '../layout/constraints/FlowCell.js';
import type { GridCellOptions } from '../layout/constraints/GridCell.js';

type TLayoutOptions = GridCellOptions & FlowCellOptions;
export default TLayoutOptions;