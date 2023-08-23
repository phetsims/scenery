// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a radial blend (essentially a chunk of a radial gradient with only a linear transition between
 * two things.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, LinearEdge, PolygonalFace, RenderColor, RenderColorSpace, RenderPath, RenderPathProgram, RenderProgram, scenery, SerializedRenderPath, SerializedRenderProgram } from '../../../imports.js';
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

export default class RenderRadialBlend extends RenderPathProgram {

  private readonly inverseTransform: Matrix3;

  public constructor(
    path: RenderPath | null,
    public readonly transform: Matrix3,
    public readonly radius0: number,
    public readonly radius1: number,
    public readonly accuracy: RenderRadialBlendAccuracy,
    public readonly zero: RenderProgram,
    public readonly one: RenderProgram,
    public readonly colorSpace: RenderColorSpace
  ) {
    assert && assert( transform.isFinite() );
    assert && assert( isFinite( radius0 ) && radius0 >= 0 );
    assert && assert( isFinite( radius1 ) && radius1 >= 0 );
    assert && assert( radius0 !== radius1 );

    super( path );

    this.inverseTransform = transform.inverted();
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderRadialBlend(
      this.getTransformedPath( transform ),
      transform.timesMatrix( this.transform ),
      this.radius0,
      this.radius1,
      this.accuracy,
      this.zero.transformed( transform ),
      this.one.transformed( transform ),
      this.colorSpace
    );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderRadialBlend &&
           this.path === other.path &&
           this.transform.equals( other.transform ) &&
           this.radius0 === other.radius0 &&
           this.radius1 === other.radius1 &&
           this.zero.equals( other.zero ) &&
           this.one.equals( other.one ) &&
           this.colorSpace === other.colorSpace;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderRadialBlend( this.path, this.transform, this.radius0, this.radius1, this.accuracy, this.zero.replace( callback ), this.one.replace( callback ), this.colorSpace );
    }
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.zero.depthFirst( callback );
    this.one.depthFirst( callback );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    return this.zero.isFullyTransparent() && this.one.isFullyTransparent();
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.zero.isFullyOpaque() && this.one.isFullyOpaque();
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const zero = this.zero.simplify( pathTest );
    const one = this.one.simplify( pathTest );

    if ( zero.isFullyTransparent() && one.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.isInPath( pathTest ) ) {
      return new RenderRadialBlend( null, this.transform, this.radius0, this.radius1, this.accuracy, zero, one, this.colorSpace );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    pathTest: ( renderPath: RenderPath ) => boolean = constantTrue
  ): Vector4 {
    if ( !this.isInPath( pathTest ) ) {
      return Vector4.ZERO;
    }

    // TODO: flag to control whether this gets set? TODO: Flag to just use centroid
    let averageDistance;
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
    assert && assert( isFinite( averageDistance ) );

    // if ( assert ) {
    //   const localPoint = scratchRadialBlendVector.set( centroid );
    //   this.inverseTransform.multiplyVector2( localPoint );
    //
    //   const maxDistance = Math.sqrt( ( maxX - minX ) ** 2 + ( maxY - minY ) ** 2 );
    //   assert( Math.abs( averageDistance - localPoint.magnitude ) < maxDistance * 5 );
    // }

    // TODO: assuming no actual order, BUT needs positive radii?
    const t = ( averageDistance - this.radius0 ) / ( this.radius1 - this.radius0 );
    assert && assert( isFinite( t ) );

    if ( t <= 0 ) {
      return this.zero.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
    }
    else if ( t >= 1 ) {
      return this.one.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
    }
    else {
      return RenderColor.ratioBlend(
        this.zero.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest ),
        this.one.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest ),
        t,
        this.colorSpace
      );
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderRadialBlend (${this.path ? this.path.id : 'null'})`;
  }

  public override serialize(): SerializedRenderRadialBlend {
    return {
      type: 'RenderRadialBlend',
      path: this.path ? this.path.serialize() : null,
      transform: [
        this.transform.m00(), this.transform.m01(), this.transform.m02(),
        this.transform.m10(), this.transform.m11(), this.transform.m12(),
        this.transform.m20(), this.transform.m21(), this.transform.m22()
      ],
      radius0: this.radius0,
      radius1: this.radius1,
      accuracy: this.accuracy,
      zero: this.zero.serialize(),
      one: this.one.serialize(),
      colorSpace: this.colorSpace
    };
  }

  public static override deserialize( obj: SerializedRenderRadialBlend ): RenderRadialBlend {
    return new RenderRadialBlend(
      obj.path ? RenderPath.deserialize( obj.path ) : null,
      Matrix3.rowMajor(
        obj.transform[ 0 ], obj.transform[ 1 ], obj.transform[ 2 ],
        obj.transform[ 3 ], obj.transform[ 4 ], obj.transform[ 5 ],
        obj.transform[ 6 ], obj.transform[ 7 ], obj.transform[ 8 ]
      ),
      obj.radius0,
      obj.radius1,
      obj.accuracy,
      RenderProgram.deserialize( obj.zero ),
      RenderProgram.deserialize( obj.one ),
      obj.colorSpace
    );
  }
}

scenery.register( 'RenderRadialBlend', RenderRadialBlend );

export type SerializedRenderRadialBlend = {
  type: 'RenderRadialBlend';
  path: SerializedRenderPath | null;
  transform: number[];
  radius0: number;
  radius1: number;
  accuracy: RenderRadialBlendAccuracy;
  zero: SerializedRenderProgram;
  one: SerializedRenderProgram;
  colorSpace: RenderColorSpace;
};
