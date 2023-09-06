// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for a classic radial gradient.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderableFace, RenderColor, RenderExtend, RenderGradientStop, RenderImage, RenderLinearRange, RenderProgram, RenderRadialBlend, RenderRadialBlendAccuracy, scenery, SerializedRenderGradientStop } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Utils from '../../../../../dot/js/Utils.js';

export enum RenderRadialGradientAccuracy {
  SplitAccurate = 0,
  SplitCentroid = 1,
  SplitPixelCenter = 2,
  UnsplitCentroid = 3,
  UnsplitPixelCenter = 4
}

scenery.register( 'RenderRadialGradientAccuracy', RenderRadialGradientAccuracy );

const scratchVectorA = new Vector2( 0, 0 );

const toProgram = ( item: RenderGradientStop ): RenderProgram => item.program;

export default class RenderRadialGradient extends RenderProgram {

  private logic: RadialGradientLogic | null = null;

  public constructor(
    public readonly transform: Matrix3,
    public readonly start: Vector2,
    public readonly startRadius: number,
    public readonly end: Vector2,
    public readonly endRadius: number,
    public readonly stops: RenderGradientStop[], // should be sorted!!
    public readonly extend: RenderExtend,
    public readonly accuracy: RenderRadialGradientAccuracy
  ) {
    assert && assert( transform.isFinite() );
    assert && assert( start.isFinite() );
    assert && assert( isFinite( startRadius ) && startRadius >= 0 );
    assert && assert( end.isFinite() );
    assert && assert( isFinite( endRadius ) && endRadius >= 0 );

    assert && assert( _.range( 0, stops.length - 1 ).every( i => {
      return stops[ i ].ratio <= stops[ i + 1 ].ratio;
    } ), 'RenderLinearGradient stops not monotonically increasing' );

    const children = stops.map( toProgram );

    super(
      children,
      _.every( children, RenderProgram.closureIsFullyTransparent ),
      _.every( children, RenderProgram.closureIsFullyOpaque ),
      false,
      false,
      accuracy === RenderRadialGradientAccuracy.UnsplitCentroid || accuracy === RenderRadialGradientAccuracy.SplitCentroid || accuracy === RenderRadialGradientAccuracy.SplitAccurate
    );
  }

  public override getName(): string {
    return 'RenderRadialGradient';
  }

  public override withChildren( children: RenderProgram[] ): RenderRadialGradient {
    assert && assert( children.length === this.stops.length );
    return new RenderRadialGradient( this.transform, this.start, this.startRadius, this.end, this.endRadius, this.stops.map( ( stop, i ) => {
      return new RenderGradientStop( stop.ratio, children[ i ] );
    } ), this.extend, this.accuracy );
  }

  public override isSplittable(): boolean {
    return this.accuracy === RenderRadialGradientAccuracy.SplitAccurate ||
           this.accuracy === RenderRadialGradientAccuracy.SplitCentroid ||
           this.accuracy === RenderRadialGradientAccuracy.SplitPixelCenter;
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderRadialGradient(
      transform.timesMatrix( this.transform ),
      this.start,
      this.startRadius,
      this.end,
      this.endRadius,
      this.stops.map( stop => new RenderGradientStop( stop.ratio, stop.program.transformed( transform ) ) ),
      this.extend,
      this.accuracy
    );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.transform.equals( other.transform ) &&
      this.start.equals( other.start ) &&
      this.startRadius === other.startRadius &&
      this.end.equals( other.end ) &&
      this.endRadius === other.endRadius &&
      this.extend === other.extend &&
      this.accuracy === other.accuracy &&
      this.stops.length === other.stops.length &&
      _.every( this.stops, ( stop, i ) => stop.ratio === other.stops[ i ].ratio );
  }

  public override getSimplified( children: RenderProgram[] ): RenderProgram | null {
    const simplifiedColorStops = this.stops.map( ( stop, i ) => stop.withProgram( children[ i ] ) );

    // TODO: compaction of triplicate stops

    if ( simplifiedColorStops.every( stop => stop.program.isFullyTransparent ) ) {
      return RenderColor.TRANSPARENT;
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
    if ( this.logic === null ) {
      this.logic = new RadialGradientLogic( this );
    }

    return this.logic.evaluate( face, area, centroid, minX, minY, maxX, maxY, this.accuracy );
  }

  public override split( face: RenderableFace ): RenderableFace[] {
    const localClippableFace = face.face.getTransformed( this.transform.inverted() );

    const blendAccuracy = this.accuracy === RenderRadialGradientAccuracy.SplitAccurate ? RenderRadialBlendAccuracy.Accurate :
                          this.accuracy === RenderRadialGradientAccuracy.SplitCentroid ? RenderRadialBlendAccuracy.Centroid :
                          RenderRadialBlendAccuracy.PixelCenter;

    const center = this.start;

    const distanceRange = localClippableFace.getDistanceRangeToInside( center );

    const isReversed = this.startRadius > this.endRadius;

    const minRadius = isReversed ? this.endRadius : this.startRadius;
    const maxRadius = isReversed ? this.startRadius : this.endRadius;
    const stops = isReversed ? this.stops.map( stop => {
      return new RenderGradientStop( 1 - stop.ratio, stop.program );
    } ).reverse() : this.stops;

    const deltaRadius = maxRadius - minRadius;
    const offset = minRadius / deltaRadius;

    const radiusToStop = ( radius: number ): number => {
      return ( radius / deltaRadius ) - offset;
    };
    const stopToRadius = ( ratio: number ): number => {
      return ( ratio + offset ) * deltaRadius;
    };

    const min = radiusToStop( distanceRange.min );
    const max = radiusToStop( distanceRange.max );

    const linearRanges = RenderLinearRange.getGradientLinearRanges( min, max, 0, this.extend, stops );

    if ( linearRanges.length < 2 ) {
      // TODO: We should be doing a replacement with a RenderRadialBlend here if possible!
      return [ face ];
    }
    else {
      const splitRadii = linearRanges.map( range => range.start ).slice( 1 ).map( stopToRadius );

      // Compute clippedFaces
      const clippedFaces: ClippableFace[] = [];
      let remainingFace = localClippableFace;
      for ( let i = 0; i < splitRadii.length; i++ ) {
        const splitRadius = splitRadii[ i ];

        // TODO: get maxAngleSplit based on magnitude!!!
        const maxAngleSplit = Math.PI / 64;

        const { insideFace, outsideFace } = remainingFace.getBinaryCircularClip( center, splitRadius, maxAngleSplit );

        clippedFaces.push( insideFace );
        remainingFace = outsideFace;
      }
      clippedFaces.push( remainingFace );

      const blendTransform = this.transform.timesMatrix( Matrix3.translation( center.x, center.y ) );

      const renderableFaces = linearRanges.map( ( range, i ) => {
        const clippedFace = clippedFaces[ i ];

        // NOTE: We need to slightly round things for later parts to work ok.
        // There result in very slight differences between vertex end points without rounding, and that is relevant
        // for the accurate clipping we do later.
        const transformedClippedFace = clippedFace.getTransformed( this.transform ).getRounded( 1e-10 );

        const replacer = ( renderProgram: RenderProgram ): RenderProgram | null => {
          if ( renderProgram !== this ) {
            return null;
          }

          if ( range.startProgram === range.endProgram ) {
            return range.startProgram.replace( replacer );
          }
          else {
            const startRadius = minRadius + range.start * deltaRadius;
            const endRadius = minRadius + range.end * deltaRadius;

            return new RenderRadialBlend(
              blendTransform,
              startRadius,
              endRadius,
              blendAccuracy,
              range.startProgram.replace( replacer ),
              range.endProgram.replace( replacer )
            );
          }
        };

        // TODO: propagate "fake" edge flags
        return new RenderableFace( transformedClippedFace, face.renderProgram.replace( replacer ).simplified(), transformedClippedFace.getBounds() );
      } ).filter( face => face.face.getArea() > 1e-8 );

      return renderableFaces;
    }
  }

  public override serialize(): SerializedRenderRadialGradient {
    return {
      type: 'RenderRadialGradient',
      transform: [
        this.transform.m00(), this.transform.m01(), this.transform.m02(),
        this.transform.m10(), this.transform.m11(), this.transform.m12(),
        this.transform.m20(), this.transform.m21(), this.transform.m22()
      ],
      start: [ this.start.x, this.start.y ],
      startRadius: this.startRadius,
      end: [ this.end.x, this.end.y ],
      endRadius: this.endRadius,
      stops: this.stops.map( stop => stop.serialize() ),
      extend: this.extend,
      accuracy: this.accuracy
    };
  }

  public static override deserialize( obj: SerializedRenderRadialGradient ): RenderRadialGradient {
    return new RenderRadialGradient(
      Matrix3.rowMajor(
        obj.transform[ 0 ], obj.transform[ 1 ], obj.transform[ 2 ],
        obj.transform[ 3 ], obj.transform[ 4 ], obj.transform[ 5 ],
        obj.transform[ 6 ], obj.transform[ 7 ], obj.transform[ 8 ]
      ),
      new Vector2( obj.start[ 0 ], obj.start[ 1 ] ),
      obj.startRadius,
      new Vector2( obj.end[ 0 ], obj.end[ 1 ] ),
      obj.endRadius,
      obj.stops.map( stop => RenderGradientStop.deserialize( stop ) ),
      obj.extend,
      obj.accuracy
    );
  }
}

const fromPoly2 = ( p0: Vector2, p1: Vector2 ): Matrix3 => {
  return Matrix3.affine(
    p1.y - p0.y, p1.x - p0.x, p0.x,
    p0.x - p1.x, p1.y - p0.y, p0.y
  );
  // TODO: remove comments once tested
  // return Transform(
  //   vec4(p1.y - p0.y, p0.x - p1.x, p1.x - p0.x, p1.y - p0.y),
  //   vec2(p0.x, p0.y)
  // );
};

const twoPointToUnitLine = ( p0: Vector2, p1: Vector2 ): Matrix3 => {
  return fromPoly2( Vector2.ZERO, Vector2.X_UNIT ).timesMatrix( fromPoly2( p0, p1 ).inverted() );
};

enum RadialGradientType {
  Circular = 1,
  Strip = 2,
  FocalOnCircle = 3,
  Cone = 4
}

class RadialGradientLogic {

  private readonly xform: Matrix3;
  private readonly focal_x: number;
  private readonly radius: number;
  private readonly kind: RadialGradientType;
  private readonly isSwapped: boolean;

  public constructor( public readonly radialGradient: RenderRadialGradient ) {
    // Two-point conical gradient based on Vello, based on https://skia.org/docs/dev/design/conical/
    let p0 = radialGradient.start;
    let p1 = radialGradient.end;
    let r0 = radialGradient.startRadius;
    let r1 = radialGradient.endRadius;

    const GRADIENT_EPSILON = 1 / ( 1 << 12 );
    const userToGradient = radialGradient.transform.inverted();

    // Output variables
    let xform: Matrix3 | null = null;
    let focal_x = 0;
    let radius = 0;
    let kind: RadialGradientType;
    let isSwapped = false;

    if ( Math.abs( r0 - r1 ) <= GRADIENT_EPSILON ) {
      // When the radii are the same, emit a strip gradient
      kind = RadialGradientType.Strip;
      const scaled = r0 / p0.distance( p1 ); // TODO: how to handle div by zero?
      xform = twoPointToUnitLine( p0, p1 ).timesMatrix( userToGradient );
      radius = scaled * scaled;
    }
    else {
      // Assume a two point conical gradient unless the centers
      // are equal.
      kind = RadialGradientType.Cone;
      if ( p0.equals( p1 ) ) {
        kind = RadialGradientType.Circular;
        // Nudge p0 a bit to avoid denormals.
        p0.addScalar( GRADIENT_EPSILON );
      }
      if ( r1 === 0 ) {
        // If r1 === 0, swap the points and radii
        isSwapped = true;
        const tmp_p = p0;
        p0 = p1;
        p1 = tmp_p;
        const tmp_r = r0;
        r0 = r1;
        r1 = tmp_r;
      }
      focal_x = r0 / ( r0 - r1 );
      const cf = p0.timesScalar( 1 - focal_x ).add( p1.timesScalar( focal_x ) );
      radius = r1 / cf.distance( p1 );
      const user_to_unit_line = twoPointToUnitLine( cf, p1 ).timesMatrix( userToGradient );
      let user_to_scaled = user_to_unit_line;
      // When r === 1, focal point is on circle
      if ( Math.abs( radius - 1 ) <= GRADIENT_EPSILON ) {
        kind = RadialGradientType.FocalOnCircle;
        const scale = 0.5 * Math.abs( 1 - focal_x );
        user_to_scaled = Matrix3.scaling( scale ).timesMatrix( user_to_unit_line );
      }
      else {
        const a = radius * radius - 1;
        const scale_ratio = Math.abs( 1 - focal_x ) / a;
        const scale_x = radius * scale_ratio;
        const scale_y = Math.sqrt( Math.abs( a ) ) * scale_ratio;
        user_to_scaled = Matrix3.scaling( scale_x, scale_y ).timesMatrix( user_to_unit_line );
      }
      xform = user_to_scaled;
    }

    this.xform = xform;
    this.focal_x = focal_x;
    this.radius = radius;
    this.kind = kind;
    this.isSwapped = isSwapped;
  }

  public evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    accuracy: RenderRadialGradientAccuracy
  ): Vector4 {
    const focal_x = this.focal_x;
    const radius = this.radius;
    const kind = this.kind;
    const is_swapped = this.isSwapped;

    // TODO: remove comments once tested
    const is_strip = kind === RadialGradientType.Strip;
    const is_circular = kind === RadialGradientType.Circular;
    const is_focal_on_circle = kind === RadialGradientType.FocalOnCircle;
    const r1_recip = is_circular ? 0 : 1 / radius;
    // let r1_recip = select(1 / radius, 0, is_circular);
    const less_scale = is_swapped || ( 1 - focal_x ) < 0 ? -1 : 1;
    // let less_scale = select(1, -1, is_swapped || (1 - focal_x) < 0);
    const t_sign = Math.sign( 1 - focal_x );

    const point = (
      accuracy === RenderRadialGradientAccuracy.UnsplitCentroid ||
      accuracy === RenderRadialGradientAccuracy.SplitCentroid ||
      accuracy === RenderRadialGradientAccuracy.SplitAccurate
    ) ? centroid : scratchVectorA.setXY( ( minX + maxX ) / 2, ( minY + maxY ) / 2 );

    // Pixel-specifics
    const local_xy = this.xform.timesVector2( point );
    const x = local_xy.x;
    const y = local_xy.y;
    const xx = x * x;
    const yy = y * y;
    let t = 0;
    let is_valid = true;
    if ( is_strip ) {
      const a = radius - yy;
      t = Math.sqrt( a ) + x;
      is_valid = a >= 0;
    }
    else if ( is_focal_on_circle ) {
      t = ( xx + yy ) / x;
      is_valid = t >= 0 && x !== 0;
    }
    else if ( radius > 1 ) {
      t = Math.sqrt( xx + yy ) - x * r1_recip;
    }
    else { // radius < 1
      const a = xx - yy;
      t = less_scale * Math.sqrt( a ) - x * r1_recip;
      is_valid = a >= 0 && t >= 0;
    }
    if ( is_valid ) {
      t = RenderImage.extend( this.radialGradient.extend, focal_x + t_sign * t );
      if ( is_swapped ) {
        t = 1 - t;
      }

      return RenderGradientStop.evaluate( face, area, centroid, minX, minY, maxX, maxY, this.radialGradient.stops, t );
    }
    else {
      // Invalid is a checkerboard red/yellow
      return ( Utils.roundSymmetric( centroid.x ) + Utils.roundSymmetric( centroid.y ) ) % 2 === 0 ? new Vector4( 1, 0, 0, 1 ) : new Vector4( 1, 1, 0, 1 );
    }
  }
}

scenery.register( 'RenderRadialGradient', RenderRadialGradient );

export type SerializedRenderRadialGradient = {
  type: 'RenderRadialGradient';
  transform: number[];
  start: number[];
  startRadius: number;
  end: number[];
  endRadius: number;
  stops: SerializedRenderGradientStop[];
  extend: RenderExtend;
  accuracy: RenderRadialGradientAccuracy;
};
