// Copyright 2021-2024, University of Colorado Boulder

import type { FlowCellOptions } from '../layout/constraints/FlowCell.js';
import type { GridCellOptions } from '../layout/constraints/GridCell.js';

/**
 * The main type interface for Node's layoutOptions (for use with Grid/Flow based layouts)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type TLayoutOptions = GridCellOptions & FlowCellOptions;
export default TLayoutOptions;