// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a linear blend (essentially a chunk of a linear gradient with only a linear transition between
 * two things.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export enum RenderLinearBlendAccuracy {
  Accurate = 0,
  PixelCenter = 1
}

scenery.register( 'RenderLinearBlendAccuracy', RenderLinearBlendAccuracy );

export default class RenderLinearBlend extends RenderProgram {

  public constructor(
    public readonly scaledNormal: Vector2,
    public readonly offset: number,
    public readonly accuracy: RenderLinearBlendAccuracy,
    public readonly zero: RenderProgram,
    public readonly one: RenderProgram
  ) {
    assert && assert( scaledNormal.isFinite() && scaledNormal.magnitude > 0 );
    assert && assert( isFinite( offset ) );

    super(
      [ zero, one ],
      zero.isFullyTransparent && one.isFullyTransparent,
      zero.isFullyOpaque && one.isFullyOpaque,
      false,
      false,
      accuracy === RenderLinearBlendAccuracy.Accurate
    );
  }

  public override getName(): string {
    return 'RenderLinearBlend';
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearBlend {
    assert && assert( children.length === 2 );
    return new RenderLinearBlend( this.scaledNormal, this.offset, this.accuracy, children[ 0 ], children[ 1 ] );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    // scaledNormal dot startPoint = offset
    // scaledNormal dot endPoint = offset + 1

    // scaledNormal dot ( offset * inverseScaledNormal ) = offset
    // scaledNormal dot ( ( offset + 1 ) * inverseScaledNormal ) = offset + 1

    const beforeStartPoint = this.scaledNormal.timesScalar( this.offset / this.scaledNormal.magnitudeSquared );
    const beforeEndPoint = this.scaledNormal.timesScalar( ( this.offset + 1 ) / this.scaledNormal.magnitudeSquared );

    const afterStartPoint = transform.timesVector2( beforeStartPoint );
    const afterEndPoint = transform.timesVector2( beforeEndPoint );
    const afterDelta = afterEndPoint.minus( afterStartPoint );

    const afterNormal = afterDelta.normalized().timesScalar( 1 / afterDelta.magnitude );
    const afterOffset = afterNormal.dot( afterStartPoint );

    assert && assert( Math.abs( afterNormal.dot( afterEndPoint ) - afterOffset - 1 ) < 1e-8, 'afterNormal.dot( afterEndPoint ) - afterOffset' );

    return new RenderLinearBlend(
      afterNormal,
      afterOffset,
      this.accuracy,
      this.zero.transformed( transform ),
      this.one.transformed( transform )
    );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.scaledNormal.equals( other.scaledNormal ) &&
           this.offset === other.offset &&
           this.accuracy === other.accuracy;
  }

  public override getSimplified( children: RenderProgram[] ): RenderProgram | null {
    const zero = children[ 0 ];
    const one = children[ 1 ];

    if ( zero.isFullyTransparent && one.isFullyTransparent ) {
      return RenderColor.TRANSPARENT;
    }
    else if ( zero.equals( one ) ) {
      return zero;
    }
    else {
      return null;
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
    const dot = this.accuracy === RenderLinearBlendAccuracy.Accurate ?
                this.scaledNormal.dot( centroid ) :
                this.scaledNormal.x * ( minX + maxX ) / 2 + this.scaledNormal.y * ( minY + maxY ) / 2;

    const t = dot - this.offset;

    if ( t <= 0 ) {
      return this.zero.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    }
    else if ( t >= 1 ) {
      return this.one.evaluate( face, area, centroid, minX, minY, maxX, maxY );
    }
    else {
      return RenderColor.ratioBlend(
        this.zero.evaluate( face, area, centroid, minX, minY, maxX, maxY ),
        this.one.evaluate( face, area, centroid, minX, minY, maxX, maxY ),
        t
      );
    }
  }

  public override serialize(): SerializedRenderLinearBlend {
    return {
      type: 'RenderLinearBlend',
      scaledNormal: [ this.scaledNormal.x, this.scaledNormal.y ],
      offset: this.offset,
      accuracy: this.accuracy,
      zero: this.zero.serialize(),
      one: this.one.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderLinearBlend ): RenderLinearBlend {
    return new RenderLinearBlend(
      new Vector2( obj.scaledNormal[ 0 ], obj.scaledNormal[ 1 ] ),
      obj.offset,
      obj.accuracy,
      RenderProgram.deserialize( obj.zero ),
      RenderProgram.deserialize( obj.one )
    );
  }
}

scenery.register( 'RenderLinearBlend', RenderLinearBlend );

export type SerializedRenderLinearBlend = {
  type: 'RenderLinearBlend';
  scaledNormal: number[];
  offset: number;
  accuracy: RenderLinearBlendAccuracy;
  zero: SerializedRenderProgram;
  one: SerializedRenderProgram;
};
