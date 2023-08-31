// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for applying a color-matrix filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, RenderUnary, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix4 from '../../../../../dot/js/Matrix4.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderFilter extends RenderUnary {
  public constructor(
    program: RenderProgram,
    public readonly colorMatrix: Matrix4,
    public readonly colorTranslation: Vector4
  ) {
    super( program );
  }

  public override getName(): string {
    return 'RenderFilter';
  }

  public override withChildren( children: RenderProgram[] ): RenderFilter {
    assert && assert( children.length === 1 );
    return new RenderFilter( children[ 0 ], this.colorMatrix, this.colorTranslation );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.colorMatrix.equals( other.colorMatrix ) &&
           this.colorTranslation.equals( other.colorTranslation );
  }

  public override simplified(): RenderProgram {
    const program = this.program.simplified();

    if ( program instanceof RenderColor ) {
      return new RenderColor( RenderFilter.applyFilter( program.color, this.colorMatrix, this.colorTranslation ) );
    }
    else if ( program !== this.program ) {
      return new RenderFilter( program, this.colorMatrix, this.colorTranslation );
    }
    else {
      return this;
    }
  }

  public override isFullyTransparent(): boolean {
    // If we modify alpha based on color value, we can't make guarantees
    if ( this.colorMatrix.m30() !== 0 || this.colorMatrix.m31() !== 0 || this.colorMatrix.m32() !== 0 ) {
      return false;
    }

    if ( this.program.isFullyTransparent() ) {
      return this.colorTranslation.w === 0;
    }
    else if ( this.program.isFullyOpaque() ) {
      return this.colorMatrix.m33() + this.colorTranslation.w === 0;
    }
    else {
      return this.colorMatrix.m33() === 0 && this.colorTranslation.w === 0;
    }
  }

  public override isFullyOpaque(): boolean {
    // If we modify alpha based on color value, we can't make guarantees
    if ( this.colorMatrix.m30() !== 0 || this.colorMatrix.m31() !== 0 || this.colorMatrix.m32() !== 0 ) {
      return false;
    }

    if ( this.program.isFullyOpaque() ) {
      return this.colorMatrix.m33() + this.colorTranslation.w === 1;
    }
    else if ( this.program.isFullyTransparent() ) {
      return this.colorTranslation.w === 1;
    }
    else {
      return this.colorMatrix.m33() === 0 && this.colorTranslation.w === 1;
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

    return RenderFilter.applyFilter( source, this.colorMatrix, this.colorTranslation );
  }

  public static applyFilter( color: Vector4, colorMatrix: Matrix4, colorTranslation: Vector4 ): Vector4 {
    return RenderColor.premultiply( colorMatrix.timesVector4( RenderColor.unpremultiply( color ) ).plus( colorTranslation ) );
  }

  public override serialize(): SerializedRenderFilter {
    return {
      type: 'RenderFilter',
      program: this.program.serialize(),
      colorMatrix: [
        this.colorMatrix.m00(), this.colorMatrix.m01(), this.colorMatrix.m02(), this.colorMatrix.m03(),
        this.colorMatrix.m10(), this.colorMatrix.m11(), this.colorMatrix.m12(), this.colorMatrix.m13(),
        this.colorMatrix.m20(), this.colorMatrix.m21(), this.colorMatrix.m22(), this.colorMatrix.m23(),
        this.colorMatrix.m30(), this.colorMatrix.m31(), this.colorMatrix.m32(), this.colorMatrix.m33()
      ],
      colorTranslation: [
        this.colorTranslation.x, this.colorTranslation.y, this.colorTranslation.z, this.colorTranslation.w
      ]
    };
  }

  public static override deserialize( obj: SerializedRenderFilter ): RenderFilter {
    return new RenderFilter(
      RenderProgram.deserialize( obj.program ),
      new Matrix4( ...obj.colorMatrix ),
      new Vector4( obj.colorTranslation[ 0 ], obj.colorTranslation[ 1 ], obj.colorTranslation[ 2 ], obj.colorTranslation[ 3 ] )
    );
  }
}

scenery.register( 'RenderFilter', RenderFilter );

export type SerializedRenderFilter = {
  type: 'RenderFilter';
  program: SerializedRenderProgram;
  colorMatrix: number[];
  colorTranslation: number[];
};
