// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for showing normals colored for debugging
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderNormalDebug extends RenderProgram {
  public constructor(
    public readonly normalProgram: RenderProgram
  ) {
    super(
      [ normalProgram ],
      false,
      false
    );
  }

  public override getName(): string {
    return 'RenderNormalDebug';
  }

  public override withChildren( children: RenderProgram[] ): RenderNormalDebug {
    assert && assert( children.length === 1 );
    return new RenderNormalDebug( children[ 0 ] );
  }

  protected override equalsTyped( other: this ): boolean {
    return true;
  }

  public override simplified(): RenderProgram {
    const normalProgram = this.normalProgram.simplified();

    if ( normalProgram.isFullyTransparent ) {
      return RenderColor.TRANSPARENT;
    }

    if ( normalProgram instanceof RenderColor ) {
      return new RenderColor( this.getDiffuse( normalProgram.color ) );
    }
    else if ( normalProgram !== this.normalProgram ) {
      return new RenderNormalDebug( normalProgram );
    }
    else {
      return this;
    }
  }

  public getDiffuse( normal: Vector4 ): Vector4 {
    assert && assert( normal.isFinite() );

    return new Vector4(
      normal.x * 0.5 + 0.5,
      normal.y * 0.5 + 0.5,
      normal.z * 0.5 + 0.5,
      1
    );
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
    const normal = this.normalProgram.evaluate( face, area, centroid, minX, minY, maxX, maxY );

    return this.getDiffuse( normal );
  }

  public override serialize(): SerializedRenderNormalDebug {
    return {
      type: 'RenderNormalDebug',
      normalProgram: this.normalProgram.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderNormalDebug ): RenderNormalDebug {
    return new RenderNormalDebug( RenderProgram.deserialize( obj.normalProgram ) );
  }
}

scenery.register( 'RenderNormalDebug', RenderNormalDebug );

export type SerializedRenderNormalDebug = {
  type: 'RenderNormalDebug';
  normalProgram: SerializedRenderProgram;
};
