// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a diffuse 3d reflection model
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderDiffuse extends RenderProgram {
  public constructor(
    // TODO: add positionProgram, because not all lights are directional
    public readonly normalProgram: RenderProgram
  ) {
    super(
      [ normalProgram ],
      false,
      false
    );
  }

  public override getName(): string {
    return 'RenderDiffuse';
  }

  public override withChildren( children: RenderProgram[] ): RenderDiffuse {
    assert && assert( children.length === 1 );
    return new RenderDiffuse( children[ 0 ] );
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
      return new RenderDiffuse( normalProgram );
    }
    else {
      return this;
    }
  }

  public getDiffuse( normal: Vector4 ): Vector4 {
    assert && assert( normal.isFinite() );

    // TODO: actual diffuse!
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

  public override serialize(): SerializedRenderDiffuse {
    return {
      type: 'RenderDiffuse',
      normalProgram: this.normalProgram.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderDiffuse ): RenderDiffuse {
    return new RenderDiffuse( RenderProgram.deserialize( obj.normalProgram ) );
  }
}

scenery.register( 'RenderDiffuse', RenderDiffuse );

export type SerializedRenderDiffuse = {
  type: 'RenderDiffuse';
  normalProgram: SerializedRenderProgram;
};
