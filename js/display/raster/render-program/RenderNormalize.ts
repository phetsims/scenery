// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for normalizing the result of another RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderNormalize extends RenderProgram {
  public constructor(
    public readonly program: RenderProgram
  ) {
    super(
      [ program ],
      program.isFullyTransparent,
      program.isFullyOpaque
    );
  }

  public override getName(): string {
    return 'RenderNormalize';
  }

  public override withChildren( children: RenderProgram[] ): RenderNormalize {
    assert && assert( children.length === 1 );
    return new RenderNormalize( children[ 0 ] );
  }

  public override simplified(): RenderProgram {
    const program = this.program.simplified();

    if ( program.isFullyTransparent ) {
      return RenderColor.TRANSPARENT;
    }

    if ( program instanceof RenderColor ) {
      return new RenderColor( program.color.magnitude > 0 ? program.color.normalized() : Vector4.ZERO );
    }
    else if ( program !== this.program ) {
      return new RenderNormalize( program );
    }
    else {
      return this;
    }
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): Vector4 {
    const source = this.program.evaluate( face, area, centroid, minX, minY, maxX, maxY );

    const magnitude = source.magnitude;
    if ( magnitude === 0 ) {
      return Vector4.ZERO;
    }
    else {
      assert && assert( source.normalized().isFinite() );

      return source.normalized();
    }
  }

  public override serialize(): SerializedRenderNormalize {
    return {
      type: 'RenderNormalize',
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderNormalize ): RenderNormalize {
    return new RenderNormalize( RenderProgram.deserialize( obj.program ) );
  }
}

scenery.register( 'RenderNormalize', RenderNormalize );

export type SerializedRenderNormalize = {
  type: 'RenderNormalize';
  program: SerializedRenderProgram;
};
