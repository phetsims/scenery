// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a radial blend (essentially a chunk of a radial gradient with only a linear transition between
 * two things.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, LinearEdge, PolygonalFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

const scratchRadialBlendVector = new Vector2( 0, 0 );

const scratchVectorA = new Vector2( 0, 0 );
const scratchVectorB = new Vector2( 0, 0 );
const scratchVectorC = new Vector2( 0, 0 );
const scratchVectorD = new Vector2( 0, 0 );

export enum RenderRadialBlendAccuracy {
  Accurate = 0,
  Centroid = 1,
  PixelCenter = 2
}

scenery.register( 'RenderRadialBlendAccuracy', RenderRadialBlendAccuracy );

export default class RenderRadialBlend extends RenderProgram {

  private readonly inverseTransform: Matrix3;

  public constructor(
    public readonly transform: Matrix3,
    public readonly radius0: number,
    public readonly radius1: number,
    public readonly accuracy: RenderRadialBlendAccuracy,
    public readonly zero: RenderProgram,
    public readonly one: RenderProgram
  ) {
    assert && assert( transform.isFinite() );
    assert && assert( isFinite( radius0 ) && radius0 >= 0 );
    assert && assert( isFinite( radius1 ) && radius1 >= 0 );
    assert && assert( radius0 !== radius1 );

    super(
      [ zero, one ],
      zero.isFullyTransparent && one.isFullyTransparent,
      zero.isFullyOpaque && one.isFullyOpaque,
      false,
      accuracy === RenderRadialBlendAccuracy.Accurate,
      accuracy === RenderRadialBlendAccuracy.Centroid
    );

    this.inverseTransform = transform.inverted();
  }

  public override getName(): string {
    return 'RenderRadialBlend';
  }

  public override withChildren( children: RenderProgram[] ): RenderRadialBlend {
    assert && assert( children.length === 2 );
    return new RenderRadialBlend( this.transform, this.radius0, this.radius1, this.accuracy, children[ 0 ], children[ 1 ] );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderRadialBlend(
      transform.timesMatrix( this.transform ),
      this.radius0,
      this.radius1,
      this.accuracy,
      this.zero.transformed( transform ),
      this.one.transformed( transform )
    );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.transform.equals( other.transform ) &&
           this.radius0 === other.radius0 &&
           this.radius1 === other.radius1 &&
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
    // TODO: flag to control whether this gets set? TODO: Flag to just use centroid
    let averageDistance;
    if ( this.accuracy === RenderRadialBlendAccuracy.Accurate ) {
      if ( face ) {
        averageDistance = face.getAverageDistanceTransformedToOrigin( this.inverseTransform, area );
      }
      else {
        // NOTE: Do the equivalent of the above, but without creating a face and a ton of garbage

        const p0 = this.inverseTransform.multiplyVector2( scratchVectorA.setXY( minX, minY ) );
        const p1 = this.inverseTransform.multiplyVector2( scratchVectorB.setXY( maxX, minY ) );
        const p2 = this.inverseTransform.multiplyVector2( scratchVectorC.setXY( maxX, maxY ) );
        const p3 = this.inverseTransform.multiplyVector2( scratchVectorD.setXY( minX, maxY ) );

        // Needs CCW orientation
        averageDistance = (
                            LinearEdge.evaluateLineIntegralDistance( p0.x, p0.y, p1.x, p1.y ) +
                            LinearEdge.evaluateLineIntegralDistance( p1.x, p1.y, p2.x, p2.y ) +
                            LinearEdge.evaluateLineIntegralDistance( p2.x, p2.y, p3.x, p3.y ) +
                            LinearEdge.evaluateLineIntegralDistance( p3.x, p3.y, p0.x, p0.y )
                          ) / ( area * this.inverseTransform.getSignedScale() );

        assert && assert( averageDistance === new PolygonalFace( [
          [
            new Vector2( minX, minY ),
            new Vector2( maxX, minY ),
            new Vector2( maxX, maxY ),
            new Vector2( minX, maxY )
          ]
        ] ).getAverageDistanceTransformedToOrigin( this.inverseTransform, area ) );
      }
    }
    else if ( this.accuracy === RenderRadialBlendAccuracy.Centroid ) {
      const localPoint = scratchRadialBlendVector.set( centroid );
      this.inverseTransform.multiplyVector2( localPoint );

      averageDistance = localPoint.magnitude;
    }
    else if ( this.accuracy === RenderRadialBlendAccuracy.PixelCenter ) {
      const localPoint = scratchRadialBlendVector.setXY( ( minX + maxX ) / 2, ( minY + maxY ) / 2 );
      this.inverseTransform.multiplyVector2( localPoint );

      averageDistance = localPoint.magnitude;
    }
    else {
      throw new Error( 'unreachable' );
    }
    assert && assert( isFinite( averageDistance ) );

    // if ( assert ) {
    //
    //   const maxDistance = Math.sqrt( ( maxX - minX ) ** 2 + ( maxY - minY ) ** 2 );
    //   assert( Math.abs( averageDistance - localPoint.magnitude ) < maxDistance * 5 );
    // }

    // TODO: assuming no actual order, BUT needs positive radii?
    const t = ( averageDistance - this.radius0 ) / ( this.radius1 - this.radius0 );
    assert && assert( isFinite( t ) );

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

  public override serialize(): SerializedRenderRadialBlend {
    return {
      type: 'RenderRadialBlend',
      transform: [
        this.transform.m00(), this.transform.m01(), this.transform.m02(),
        this.transform.m10(), this.transform.m11(), this.transform.m12(),
        this.transform.m20(), this.transform.m21(), this.transform.m22()
      ],
      radius0: this.radius0,
      radius1: this.radius1,
      accuracy: this.accuracy,
      zero: this.zero.serialize(),
      one: this.one.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderRadialBlend ): RenderRadialBlend {
    return new RenderRadialBlend(
      Matrix3.rowMajor(
        obj.transform[ 0 ], obj.transform[ 1 ], obj.transform[ 2 ],
        obj.transform[ 3 ], obj.transform[ 4 ], obj.transform[ 5 ],
        obj.transform[ 6 ], obj.transform[ 7 ], obj.transform[ 8 ]
      ),
      obj.radius0,
      obj.radius1,
      obj.accuracy,
      RenderProgram.deserialize( obj.zero ),
      RenderProgram.deserialize( obj.one )
    );
  }
}

scenery.register( 'RenderRadialBlend', RenderRadialBlend );

export type SerializedRenderRadialBlend = {
  type: 'RenderRadialBlend';
  transform: number[];
  radius0: number;
  radius1: number;
  accuracy: RenderRadialBlendAccuracy;
  zero: SerializedRenderProgram;
  one: SerializedRenderProgram;
};
