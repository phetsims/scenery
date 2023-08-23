// Copyright 2023, University of Colorado Boulder

/**
 * Stores information about what a RenderProgram needs in order to be evaluated
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

export default class RenderProgramNeeds {
  public constructor(
    public readonly needsFace: boolean,
    public readonly needsArea: boolean,
    public readonly needsCentroid: boolean
  ) {}
}

scenery.register( 'RenderProgramNeeds', RenderProgramNeeds );
