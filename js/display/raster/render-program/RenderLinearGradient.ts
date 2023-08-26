// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a classic linear gradient
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderExtend, RenderGradientStop, RenderImage, RenderPath, RenderProgram, scenery, SerializedRenderGradientStop } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export enum RenderLinearGradientAccuracy {
  SplitAccurate = 0,
  SplitPixelCenter = 2,
  UnsplitCentroid = 3,
  UnsplitPixelCenter = 4
}

scenery.register( 'RenderLinearGradientAccuracy', RenderLinearGradientAccuracy );

const scratchLinearGradientVector0 = new Vector2( 0, 0 );

export default class RenderLinearGradient extends RenderProgram {

  public readonly inverseTransform: Matrix3;
  private readonly isIdentity: boolean;
  private readonly gradDelta: Vector2;

  public constructor(
    public readonly transform: Matrix3,
    public readonly start: Vector2,
    public readonly end: Vector2,
    public readonly stops: RenderGradientStop[], // should be sorted!!
    public readonly extend: RenderExtend,
    public readonly accuracy: RenderLinearGradientAccuracy
  ) {
    assert && assert( transform.isFinite() );
    assert && assert( start.isFinite() );
    assert && assert( end.isFinite() );
    assert && assert( !start.equals( end ) );

    assert && assert( _.range( 0, stops.length - 1 ).every( i => {
      return stops[ i ].ratio <= stops[ i + 1 ].ratio;
    } ), 'RenderLinearGradient stops not monotonically increasing' );

    super();

    this.inverseTransform = transform.inverted();
    this.isIdentity = transform.isIdentity();
    this.gradDelta = end.minus( start );
  }

  public override getChildren(): RenderProgram[] {
    return this.stops.map( stop => stop.program );
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearGradient {
    assert && assert( children.length === this.stops.length );
    return new RenderLinearGradient( this.transform, this.start, this.end, this.stops.map( ( stop, i ) => {
      return new RenderGradientStop( stop.ratio, children[ i ] );
    } ), this.extend, this.accuracy );
  }

  public isSplittable(): boolean {
    return this.accuracy === RenderLinearGradientAccuracy.SplitAccurate ||
           this.accuracy === RenderLinearGradientAccuracy.SplitPixelCenter;
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderLinearGradient(
      transform.timesMatrix( this.transform ),
      this.start,
      this.end,
      this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.transformed( transform ) ) ),
      this.extend,
      this.accuracy
    );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderLinearGradient &&
      this.transform.equals( other.transform ) &&
      this.start.equals( other.start ) &&
      this.end.equals( other.end ) &&
      this.stops.length === other.stops.length &&
      // TODO perf
      this.stops.every( ( stop, i ) => stop.ratio === other.stops[ i ].ratio && stop.program.equals( other.stops[ i ].program ) ) &&
      this.extend === other.extend &&
      this.accuracy === other.accuracy;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderLinearGradient(
        this.transform,
        this.start,
        this.end,
        this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.replace( callback ) ) ),
        this.extend,
        this.accuracy
      );
    }
  }

  public override isFullyTransparent(): boolean {
    return this.stops.every( stop => stop.program.isFullyTransparent() );
  }

  public override isFullyOpaque(): boolean {
    return this.stops.every( stop => stop.program.isFullyOpaque() );
  }

  public override needsFace(): boolean {
    for ( let i = 0; i < this.stops.length; i++ ) {
      if ( this.stops[ i ].program.needsFace() ) {
        return true;
      }
    }
    return false;
  }

  public override needsArea(): boolean {
    for ( let i = 0; i < this.stops.length; i++ ) {
      if ( this.stops[ i ].program.needsArea() ) {
        return true;
      }
    }
    return false;
  }

  public override needsCentroid(): boolean {
    if ( this.useInternalCentroid() ) {
      return true;
    }
    for ( let i = 0; i < this.stops.length; i++ ) {
      if ( this.stops[ i ].program.needsCentroid() ) {
        return true;
      }
    }
    return false;
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const simplifiedColorStops = this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.simplify( pathTest ) ) );

    if ( simplifiedColorStops.every( stop => stop.program.isFullyTransparent() ) ) {
      return RenderColor.TRANSPARENT;
    }

    return new RenderLinearGradient( this.transform, this.start, this.end, simplifiedColorStops, this.extend, this.accuracy );
  }

  private useInternalCentroid(): boolean {
    return this.accuracy === RenderLinearGradientAccuracy.UnsplitCentroid || this.accuracy === RenderLinearGradientAccuracy.SplitAccurate;
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
    const point = this.useInternalCentroid() ?
                  scratchLinearGradientVector0.set( centroid ) :
                  scratchLinearGradientVector0.setXY( ( minX + maxX ) / 2, ( minY + maxY ) / 2 );

    const localPoint = point;
    if ( !this.isIdentity ) {
      this.inverseTransform.multiplyVector2( localPoint );
    }

    const localDelta = localPoint.subtract( this.start ); // MUTABLE, changes localPoint
    const gradDelta = this.gradDelta;

    const t = gradDelta.magnitude > 0 ? localDelta.dot( gradDelta ) / gradDelta.dot( gradDelta ) : 0;
    const mappedT = RenderImage.extend( this.extend, t );

    return RenderGradientStop.evaluate( face, area, centroid, minX, minY, maxX, maxY, this.stops, mappedT, pathTest );
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderLinearGradient`;
  }

  public override serialize(): SerializedRenderLinearGradient {
    return {
      type: 'RenderLinearGradient',
      transform: [
        this.transform.m00(), this.transform.m01(), this.transform.m02(),
        this.transform.m10(), this.transform.m11(), this.transform.m12(),
        this.transform.m20(), this.transform.m21(), this.transform.m22()
      ],
      start: [ this.start.x, this.start.y ],
      end: [ this.end.x, this.end.y ],
      stops: this.stops.map( stop => stop.serialize() ),
      extend: this.extend,
      accuracy: this.accuracy
    };
  }

  public static override deserialize( obj: SerializedRenderLinearGradient ): RenderLinearGradient {
    return new RenderLinearGradient(
      Matrix3.rowMajor(
        obj.transform[ 0 ], obj.transform[ 1 ], obj.transform[ 2 ],
        obj.transform[ 3 ], obj.transform[ 4 ], obj.transform[ 5 ],
        obj.transform[ 6 ], obj.transform[ 7 ], obj.transform[ 8 ]
      ),
      new Vector2( obj.start[ 0 ], obj.start[ 1 ] ),
      new Vector2( obj.end[ 0 ], obj.end[ 1 ] ),
      obj.stops.map( stop => RenderGradientStop.deserialize( stop ) ),
      obj.extend,
      obj.accuracy
    );
  }
}

scenery.register( 'RenderLinearGradient', RenderLinearGradient );

export type SerializedRenderLinearGradient = {
  type: 'RenderLinearGradient';
  transform: number[];
  start: number[];
  end: number[];
  stops: SerializedRenderGradientStop[];
  extend: RenderExtend;
  accuracy: RenderLinearGradientAccuracy;
};
