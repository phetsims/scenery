// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a classic linear gradient
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderColorSpace, RenderExtend, RenderGradientStop, RenderImage, RenderPath, RenderPathProgram, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

const scratchLinearGradientVector0 = new Vector2( 0, 0 );

export default class RenderLinearGradient extends RenderPathProgram {

  public readonly inverseTransform: Matrix3;
  private readonly isIdentity: boolean;
  private readonly gradDelta: Vector2;

  public constructor(
    path: RenderPath | null,
    public readonly transform: Matrix3,
    public readonly start: Vector2,
    public readonly end: Vector2,
    public readonly stops: RenderGradientStop[], // should be sorted!!
    public readonly extend: RenderExtend,
    public readonly colorSpace: RenderColorSpace
  ) {
    assert && assert( transform.isFinite() );
    assert && assert( start.isFinite() );
    assert && assert( end.isFinite() );
    assert && assert( !start.equals( end ) );

    assert && assert( _.range( 0, stops.length - 1 ).every( i => {
      return stops[ i ].ratio <= stops[ i + 1 ].ratio;
    } ), 'RenderLinearGradient stops not monotonically increasing' );

    super( path );

    this.inverseTransform = transform.inverted();
    this.isIdentity = transform.isIdentity();
    this.gradDelta = end.minus( start );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderLinearGradient(
      this.getTransformedPath( transform ),
      transform.timesMatrix( this.transform ),
      this.start,
      this.end,
      this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.transformed( transform ) ) ),
      this.extend,
      this.colorSpace
    );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
      other instanceof RenderLinearGradient &&
      this.transform.equals( other.transform ) &&
      this.start.equals( other.start ) &&
      this.end.equals( other.end ) &&
      this.stops.length === other.stops.length &&
      // TODO perf
      this.stops.every( ( stop, i ) => stop.ratio === other.stops[ i ].ratio && stop.program.equals( other.stops[ i ].program ) ) &&
      this.extend === other.extend &&
      this.colorSpace === other.colorSpace;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderLinearGradient(
        this.path,
        this.transform,
        this.start,
        this.end,
        this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.replace( callback ) ) ),
        this.extend,
        this.colorSpace
      );
    }
  }

  public override depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.stops.forEach( stop => stop.program.depthFirst( callback ) );
    callback( this );
  }

  public override isFullyTransparent(): boolean {
    return this.stops.every( stop => stop.program.isFullyTransparent() );
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.stops.every( stop => stop.program.isFullyOpaque() );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const simplifiedColorStops = this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.simplify( pathTest ) ) );

    if ( simplifiedColorStops.every( stop => stop.program.isFullyTransparent() ) ) {
      return RenderColor.TRANSPARENT;
    }

    if ( this.isInPath( pathTest ) ) {
      return new RenderLinearGradient( null, this.transform, this.start, this.end, simplifiedColorStops, this.extend, this.colorSpace );
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

    const localPoint = scratchLinearGradientVector0.set( centroid );
    if ( !this.isIdentity ) {
      this.inverseTransform.multiplyVector2( localPoint );
    }

    const localDelta = localPoint.subtract( this.start ); // MUTABLE, changes localPoint
    const gradDelta = this.gradDelta;

    const t = gradDelta.magnitude > 0 ? localDelta.dot( gradDelta ) / gradDelta.dot( gradDelta ) : 0;
    const mappedT = RenderImage.extend( this.extend, t );

    return RenderGradientStop.evaluate( face, area, centroid, minX, minY, maxX, maxY, this.stops, mappedT, this.colorSpace, pathTest );
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderLinearGradient (${this.path ? this.path.id : 'null'})`;
  }
}

scenery.register( 'RenderLinearGradient', RenderLinearGradient );
