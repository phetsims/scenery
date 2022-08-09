// Copyright 2021-2022, University of Colorado Boulder

import { FlowCellOptions, GridCellOptions } from '../imports.js';

/**
 * The main type interface for Node's layoutOptions (for use with Grid/Flow based layouts)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export type TLayoutOptions = GridCellOptions & FlowCellOptions;
export default TLayoutOptions;
