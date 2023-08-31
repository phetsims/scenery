// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a RenderProgram/bounds, that can potentially be split into multiples, or optimized
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderProgram, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';

export default class RenderableFace {
  public constructor(
    public readonly face: ClippableFace,
    public readonly renderProgram: RenderProgram,
    public readonly bounds: Bounds2
  ) {}

  // TODO: FIND A BETTER STRATEGY rather than multiple full iteration for every RenderProgram that gets split!!!!
  public split(): RenderableFace[] {
    // We're going to record which programs are fully split. If they are still present in the tree, we'll know we
    // already split them.
    const fullySplitRenderPrograms = new Set<RenderProgram>();

    let faces: RenderableFace[] = [ this ];

    const getNextSplittable = ( faces: RenderableFace[] ): RenderProgram | null => {

      for ( let i = 0; i < faces.length; i++ ) {
        const face = faces[ i ];

        let result: RenderProgram | null = null;

        face.renderProgram.depthFirst( subProgram => {
          // TODO: early exit?
          if ( !fullySplitRenderPrograms.has( subProgram ) && subProgram.isSplittable() ) {
            result = subProgram;
          }
        } );

        if ( result ) {
          return result;
        }
      }

      return null;
    };

    let splitProgram: RenderProgram | null;

    // eslint-disable-next-line no-cond-assign
    while ( splitProgram = getNextSplittable( faces ) ) {
      const newFaces: RenderableFace[] = [];

      for ( let i = 0; i < faces.length; i++ ) {
        const face = faces[ i ];

        if ( face.renderProgram.containsRenderProgram( splitProgram ) ) {
          newFaces.push( ...splitProgram.split( face ) );
        }
        else {
          newFaces.push( face );
        }
      }

      fullySplitRenderPrograms.add( splitProgram );

      faces = newFaces;
    }

    return faces;
  }
}

scenery.register( 'RenderableFace', RenderableFace );
