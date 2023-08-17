// Copyright 2023, University of Colorado Boulder

/**
 * A constant-true function for use as a default for RenderProgram simplification
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderPath, scenery } from '../../../imports.js';

const constantTrue: ( renderPath: RenderPath ) => boolean = _.constant( true );

export default constantTrue;

scenery.register( 'constantTrue', constantTrue );
