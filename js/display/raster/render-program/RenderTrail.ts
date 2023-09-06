// Copyright 2023, University of Colorado Boulder

/**
 * Represents an ancestor-to-descendant walk through a RenderProgram tree.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderProgram, scenery } from '../../../imports.js';

export default class RenderTrail {
  public constructor(
    public readonly program: RenderProgram,
    public readonly indices: number[]
  ) {}

  public equals( other: RenderTrail ): boolean {
    return this.indices.length === other.indices.length && this.compare( other ) === 0;
  }

  public compare( other: RenderTrail ): number {
    const minIndexLength = Math.min( this.indices.length, other.indices.length );

    for ( let i = 0; i < minIndexLength; i++ ) {
      const index = this.indices[ i ];
      const otherIndex = other.indices[ i ];

      if ( index !== otherIndex ) {
        return Math.sign( index - otherIndex );
      }
    }

    return this.indices.length - other.indices.length;
  }

  public static closureCompare( a: RenderTrail, b: RenderTrail ): number {
    return a.compare( b );
  }
}

scenery.register( 'RenderTrail', RenderTrail );
