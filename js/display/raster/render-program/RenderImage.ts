// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for an image
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, PolygonMitchellNetravali, RenderExtend, RenderImageable, RenderPath, RenderProgram, RenderResampleType, scenery, SerializedRenderImageable } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Utils from '../../../../../dot/js/Utils.js';

export default class RenderImage extends RenderProgram {

  public readonly inverseTransform: Matrix3;
  public readonly inverseTransformWithHalfOffset: Matrix3;

  // TODO: Support mipmaps
  // TODO: For mipmaps, see if our approach is ideal. See Gaussian pyramid, blur before subsampling (ours might be ideal?)
  public constructor(
    public readonly transform: Matrix3,
    public readonly image: RenderImageable,
    public readonly extendX: RenderExtend,
    public readonly extendY: RenderExtend,
    public readonly resampleType: RenderResampleType
  ) {
    super();

    this.inverseTransform = transform.inverted();
    this.inverseTransformWithHalfOffset = Matrix3.translation( -0.5, -0.5 ).timesMatrix( this.inverseTransform );
  }

  public override getName(): string {
    return 'RenderImage';
  }

  public override getChildren(): RenderProgram[] {
    return [];
  }

  public override withChildren( children: RenderProgram[] ): RenderImage {
    assert && assert( children.length === 0 );
    return this;
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderImage( transform.timesMatrix( this.transform ), this.image, this.extendX, this.extendY, this.resampleType );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderImage &&
      this.transform.equals( other.transform ) &&
      this.image === other.image &&
      this.extendX === other.extendX &&
      this.extendY === other.extendY;
  }

  public override isFullyTransparent(): boolean {
    return false;
  }

  public override isFullyOpaque(): boolean {
    return this.image.isFullyOpaque;
  }

  public override needsFace(): boolean {
    return this.resampleType === RenderResampleType.AnalyticBox ||
           this.resampleType === RenderResampleType.AnalyticBilinear ||
           this.resampleType === RenderResampleType.AnalyticMitchellNetravali;
  }

  public override needsArea(): boolean {
    return false;
  }

  public override needsCentroid(): boolean {
    // TODO: Consider NOT getting centroid for some filtering, as a performance boost
    return this.resampleType === RenderResampleType.NearestNeighbor ||
           this.resampleType === RenderResampleType.Bilinear ||
           this.resampleType === RenderResampleType.MitchellNetravali;
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    return this;
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
    // TODO: analytic box! Bilinear! Bicubic! (can we mipmap for those?)
    switch( this.resampleType ) {
      case RenderResampleType.NearestNeighbor: {
        const localPoint = this.inverseTransformWithHalfOffset.timesVector2( centroid );
        const roundedX = Utils.roundSymmetric( localPoint.x );
        const roundedY = Utils.roundSymmetric( localPoint.y );
        const x = RenderImage.extendInteger( roundedX, this.image.width, this.extendX );
        const y = RenderImage.extendInteger( roundedY, this.image.height, this.extendY );

        return this.image.evaluate( x, y );
      }
      case RenderResampleType.Bilinear: {
        const localPoint = this.inverseTransformWithHalfOffset.timesVector2( centroid );

        const floorX = Math.floor( localPoint.x );
        const floorY = Math.floor( localPoint.y );
        const ceilX = Math.ceil( localPoint.x );
        const ceilY = Math.ceil( localPoint.y );

        const minX = RenderImage.extendInteger( floorX, this.image.width, this.extendX );
        const minY = RenderImage.extendInteger( floorY, this.image.height, this.extendY );
        const maxX = RenderImage.extendInteger( ceilX, this.image.width, this.extendX );
        const maxY = RenderImage.extendInteger( ceilY, this.image.height, this.extendY );

        const fractionX = localPoint.x - floorX;
        const fractionY = localPoint.y - floorY;

        const a = this.image.evaluate( minX, minY );
        const b = this.image.evaluate( minX, maxY );
        const c = this.image.evaluate( maxX, minY );
        const d = this.image.evaluate( maxX, maxY );

        // TODO: allocation reduction?
        const ab = a.timesScalar( 1 - fractionY ).plus( b.timesScalar( fractionY ) );
        const cd = c.timesScalar( 1 - fractionY ).plus( d.timesScalar( fractionY ) );
        return ab.timesScalar( 1 - fractionX ).plus( cd.timesScalar( fractionX ) );
      }
      case RenderResampleType.MitchellNetravali: {
        const localPoint = this.inverseTransformWithHalfOffset.timesVector2( centroid );

        const floorX = Math.floor( localPoint.x );
        const floorY = Math.floor( localPoint.y );

        const px0 = floorX - 1;
        const px1 = floorX;
        const px2 = floorX + 1;
        const px3 = floorX + 2;
        const py0 = floorY - 1;
        const py1 = floorY;
        const py2 = floorY + 1;
        const py3 = floorY + 2;

        const x0 = RenderImage.extendInteger( px0, this.image.width, this.extendX );
        const x1 = RenderImage.extendInteger( px1, this.image.width, this.extendX );
        const x2 = RenderImage.extendInteger( px2, this.image.width, this.extendX );
        const x3 = RenderImage.extendInteger( px3, this.image.width, this.extendX );
        const y0 = RenderImage.extendInteger( py0, this.image.height, this.extendY );
        const y1 = RenderImage.extendInteger( py1, this.image.height, this.extendY );
        const y2 = RenderImage.extendInteger( py2, this.image.height, this.extendY );
        const y3 = RenderImage.extendInteger( py3, this.image.height, this.extendY );

        const filterX0 = PolygonMitchellNetravali.evaluateFilter( localPoint.x - px0 );
        const filterX1 = PolygonMitchellNetravali.evaluateFilter( localPoint.x - px1 );
        const filterX2 = PolygonMitchellNetravali.evaluateFilter( localPoint.x - px2 );
        const filterX3 = PolygonMitchellNetravali.evaluateFilter( localPoint.x - px3 );
        const filterY0 = PolygonMitchellNetravali.evaluateFilter( localPoint.y - py0 );
        const filterY1 = PolygonMitchellNetravali.evaluateFilter( localPoint.y - py1 );
        const filterY2 = PolygonMitchellNetravali.evaluateFilter( localPoint.y - py2 );
        const filterY3 = PolygonMitchellNetravali.evaluateFilter( localPoint.y - py3 );

        const color = Vector4.ZERO.copy();

        // TODO: allocation reduction?
        color.add( this.image.evaluate( x0, y0 ).timesScalar( filterX0 * filterY0 ) );
        color.add( this.image.evaluate( x0, y1 ).timesScalar( filterX0 * filterY1 ) );
        color.add( this.image.evaluate( x0, y2 ).timesScalar( filterX0 * filterY2 ) );
        color.add( this.image.evaluate( x0, y3 ).timesScalar( filterX0 * filterY3 ) );
        color.add( this.image.evaluate( x1, y0 ).timesScalar( filterX1 * filterY0 ) );
        color.add( this.image.evaluate( x1, y1 ).timesScalar( filterX1 * filterY1 ) );
        color.add( this.image.evaluate( x1, y2 ).timesScalar( filterX1 * filterY2 ) );
        color.add( this.image.evaluate( x1, y3 ).timesScalar( filterX1 * filterY3 ) );
        color.add( this.image.evaluate( x2, y0 ).timesScalar( filterX2 * filterY0 ) );
        color.add( this.image.evaluate( x2, y1 ).timesScalar( filterX2 * filterY1 ) );
        color.add( this.image.evaluate( x2, y2 ).timesScalar( filterX2 * filterY2 ) );
        color.add( this.image.evaluate( x2, y3 ).timesScalar( filterX2 * filterY3 ) );
        color.add( this.image.evaluate( x3, y0 ).timesScalar( filterX3 * filterY0 ) );
        color.add( this.image.evaluate( x3, y1 ).timesScalar( filterX3 * filterY1 ) );
        color.add( this.image.evaluate( x3, y2 ).timesScalar( filterX3 * filterY2 ) );
        color.add( this.image.evaluate( x3, y3 ).timesScalar( filterX3 * filterY3 ) );

        return color;
      }
      case RenderResampleType.AnalyticBox: {
        return RenderImage.evaluateAnalyticFilter(
          this, face,
          minX, minY, maxX, maxY,
          this.inverseTransform,
          0, 1, -1,
          ( face: ClippableFace, x: number, y: number, px: number, py: number, area: number ) => {
            return area;
          },
          _.constant( 1 )
        );
      }
      case RenderResampleType.AnalyticBilinear: {
        return RenderImage.evaluateAnalyticFilter(
          this, face,
          minX, minY, maxX, maxY,
          this.inverseTransformWithHalfOffset,
          1, 1, 0,
          ( face: ClippableFace, x: number, y: number, px: number, py: number, area: number ) => {
            return face.getBilinearFiltered( x, y, px, py );
          },
          _.constant( 0.25 )
        );
      }
      case RenderResampleType.AnalyticMitchellNetravali: {
        return RenderImage.evaluateAnalyticFilter(
          this, face,
          minX, minY, maxX, maxY,
          this.inverseTransformWithHalfOffset,
          2, 2, 0,
          ( face: ClippableFace, x: number, y: number, px: number, py: number, area: number ) => {
            return face.getMitchellNetravaliFiltered( x, y, px, py );
          },
          PolygonMitchellNetravali.evaluateFull
        );
      }
      default:
        throw new Error( 'unknown resample type: ' + this.resampleType );
    }
  }

  /**
   * Evaluates a section of a filter analytically.
   *
   * Basically, we transform the shape (that is presumably clipped to a user-space "pixel") into our image's
   * coordinate frame. It might overlap one image-coordinate-frame pixel, or many. We'll figure out which pixels it
   * overlaps, clip it to each of those individually, and then we'll evaluate the filter for each of those.
   *
   * For most filters, the contribution of a single image-pixel (to our user-pixel) will then be determined by
   * convolving the filter with the clipped shape, and summing up the integral. For our filters, we're able to
   * use Green's theorem to evaluate this integral analytically, by doing a line integral along the edges of the
   * indvidual areas.
   *
   * The filters we are using have piecewise-polynomial functions, and thus their line integrals are similiarly
   * "split in pieces". So for e.g. bilinear, we need to evaluate a 2x2 grid of "clipped" image-pixels individually to
   * figure out the contribution. This means we need to get the clipped path for each image-pixel, and then we evaluate
   * a specific line integral on the outline of that (and sum it up for each image-pixel in the grid).
   *
   * We'll need to divide out the image-space result by the image-space area at the end
   *
   * NOTE: we might have a flipped transform, so our signed area might be NEGATIVE in this case, so this code handles
   * positive or negative areas.
   *
   * NOTE: some filters can result in negative contributions for some pixels, so we need to handle that too!
   *
   * @param renderImage
   * @param face
   * @param minX
   * @param minY
   * @param maxX
   * @param maxY
   * @param inverseTransform - The transform to put the face within the image's coordinate frame
   * @param minExpand - How far our filter expands to the min direction (left/top) - 0 box, 1 bilinear, 2 mitchell
   * @param maxExpand - How far our filter expands to the max direction (right/bottom) - 1 box, 1 bilinear, 2 mitchell
   * @param boundsShift - A shift of indices, should be -1 for the box filter (since it is offset)
   * @param evaluateFace - Evaluation function for a partial pixel. Evaluates the filter centered at x,y with the
   * bounds of the clipped pixel in the range of px,py,px+1,py+1
   * @param evaluateFull - Evaluation function for a full pixel. Evaluates the filter centered at x,y with a full
   * pixel of the bounds px,py,px+1,py+1
   */
  public static evaluateAnalyticFilter(
    renderImage: RenderImage,
    face: ClippableFace | null,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    inverseTransform: Matrix3,
    minExpand: number,
    maxExpand: number,
    boundsShift: number,
    evaluateFace: ( face: ClippableFace, x: number, y: number, px: number, py: number, area: number ) => number,
    evaluateFull: ( x: number, y: number, px: number, py: number ) => number
  ): Vector4 {

    // TODO: binary clipping will likely help prevent the "normal" filtering artifacts we were seeing.
    // TODO: The "fake" points that are far away blow up the numerical precision of the results.
    // TODO: Try to follow our normal rasterization setup.

    // TODO: Can we fo with the binary clipping, and avoid computing sections that we don't need?

    // TODO: perhaps switch to our binary clipping, because we could potentially get massive speed ups by determining
    // TODO: which regions our full "grid" is included inside (so we could just sum the pixel)

    // If we don't have a face (i.e. we are taking up the full bounds specified by minX/minY/maxX/maxY), we'll
    // construct a face that covers the entire bounds.
    face = RenderProgram.ensureFace( face, minX, minY, maxX, maxY );

    // We'll mutate and return this
    const color = Vector4.ZERO.copy();

    // Such that 0,0 now aligns with the center of our 0,0 pixel sample, and is scaled so that pixel samples are
    // at every integer coordinate pair. (or in the case of the box filter, we have an offset).
    const localFace = face.getTransformed( inverseTransform );

    const localBounds = localFace.getBounds().roundedOut();
    assert && assert( localBounds.minX < localBounds.maxX );
    assert && assert( localBounds.minY < localBounds.maxY );

    // Splits for our image-space pixels
    const horizontalSplitValues = _.range( localBounds.minX + 1, localBounds.maxX );
    const verticalSplitValues = _.range( localBounds.minY + 1, localBounds.maxY );
    const horizontalCount = horizontalSplitValues.length + 1;
    const verticalCount = verticalSplitValues.length + 1;

    // TODO: GRID clip, OR even more optimized stripe clips? (or wait... do we not burn much by stripe clipping unused regions?)
    const rows = verticalSplitValues.length ? localFace.getStripeLineClip( Vector2.Y_UNIT, verticalSplitValues, ( minX + maxX ) / 2 ) : [ localFace ];

    assertSlow && assertSlow( Math.abs( localFace.getArea() - _.sum( rows.map( f => f.getArea() ) ) ) < 1e-6 );

    const areas: number[][] = [];
    const pixelFaces = rows.map( face => {
      const row = horizontalSplitValues.length ? face.getStripeLineClip( Vector2.X_UNIT, horizontalSplitValues, ( minY + maxY ) / 2 ) : [ face ];

      assertSlow && assertSlow( Math.abs( face.getArea() - _.sum( row.map( f => f.getArea() ) ) ) < 1e-6 );

      areas.push( row.map( face => face.getArea() ) );
      return row;
    } );

    const localIndexMin = minExpand + boundsShift + 1;
    const localIndexMax = maxExpand + boundsShift;

    // box: -0, +0
    // bilinear: 0, +1
    // mitchell: -1, +2
    const iterMinX = localBounds.minX - localIndexMin;
    const iterMinY = localBounds.minY - localIndexMin;
    const iterMaxX = localBounds.maxX + localIndexMax;
    const iterMaxY = localBounds.maxY + localIndexMax;

    // For each image-space pixel whose evaluation grid will overlap with our shape's image-space pixel coverage
    for ( let y = iterMinY; y < iterMaxY; y++ ) {
      const mappedY = RenderImage.extendInteger( y, renderImage.image.height, renderImage.extendY );

      // box: -0, +1
      // bilinear: -1, +1
      // mitchell: -2, +2
      const subIterMinY = y - minExpand;
      const subIterMaxY = y + maxExpand;

      for ( let x = iterMinX; x < iterMaxX; x++ ) {
        const mappedX = RenderImage.extendInteger( x, renderImage.image.width, renderImage.extendX );
        let contribution = 0;

        const subIterMinX = x - minExpand;
        const subIterMaxX = x + maxExpand;

        // For each image-space pixel in that grid
        for ( let py = subIterMinY; py < subIterMaxY; py++ ) {
          const yIndex = py - localBounds.minY;

          if ( yIndex >= 0 && yIndex < verticalCount ) {
            for ( let px = subIterMinX; px < subIterMaxX; px++ ) {
              const xIndex = px - localBounds.minX;

              if ( xIndex >= 0 && xIndex < horizontalCount ) {
                const pixelArea = areas[ yIndex ][ xIndex ];

                const absPixelArea = Math.abs( pixelArea );

                // If it has zero area, it won't have contribution
                if ( absPixelArea > 1e-8 ) {

                  // If it has a full pixel of area, we can simplify computation SIGNIFICANTLY
                  if ( absPixelArea > 1 - 1e-8 ) {
                    contribution += Math.sign( pixelArea ) * evaluateFull( x, y, px, py );
                  }
                  else {
                    contribution += evaluateFace( pixelFaces[ yIndex ][ xIndex ], x, y, px, py, pixelArea );
                  }
                }
              }
            }
          }
        }

        if ( Math.abs( contribution ) > 1e-8 ) {
          const imageColor = renderImage.image.evaluate( mappedX, mappedY );
          color.add( imageColor.timesScalar( contribution ) );
        }
      }
    }

    // NOTE: this might flip the sign back to positive of the color (if our transform flipped the orientation)
    if ( renderImage.image.isFullyOpaque ) {
      // Our precision is actually... not great with these equations.
      assert && assert( !renderImage.image.isFullyOpaque || ( color.w / localFace.getArea() >= 1 - 1e-2 ) );

      // We can get an exact alpha here
      color.multiplyScalar( 1 / color.w );
    }
    else {
      color.multiplyScalar( 1 / localFace.getArea() );
    }

    return color;
  }

  public static extend( extend: RenderExtend, t: number ): number {
    switch( extend ) {
      case RenderExtend.Pad:
        return Utils.clamp( t, 0, 1 );
      case RenderExtend.Repeat:
        return t - Math.floor( t );
      case RenderExtend.Reflect:
        return Math.abs( t - 2.0 * Utils.roundSymmetric( 0.5 * t ) );
        // return ( Math.floor( t ) % 2 === 0 ? t : 1 - t ) - Math.floor( t );
      default:
        throw new Error( 'Unknown RenderExtend' );
    }
  }

  // Integer version of extend_mode.
  // Given size=4, provide the following patterns:
  //
  // input:  -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
  //
  // pad:     0,  0,  0,  0,  0,  0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3
  // repeat:  2,  3,  0,  1,  2,  3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1
  // reflect: 2,  3,  3,  2,  1,  0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1
  public static extendInteger( i: number, size: number, extend: RenderExtend ): number {
    switch( extend ) {
      case RenderExtend.Pad: {
        return Utils.clamp( i, 0, size - 1 );
      }
      case RenderExtend.Repeat: {
        if ( i >= 0 ) {
          return i % size;
        }
        else {
          return size - ( ( -i - 1 ) % size ) - 1;
        }
      }
      case RenderExtend.Reflect: {
        // easier to convert both to positive (with a repeat offset)
        const positiveI = i < 0 ? -i - 1 : i;

        const section = positiveI % ( size * 2 );
        if ( section < size ) {
          return section;
        }
        else {
          return 2 * size - section - 1;
        }
      }
      default: {
        throw new Error( 'Unknown RenderExtend' );
      }
    }
  }

  public override serialize(): SerializedRenderImage {
    return {
      type: 'RenderImage',
      transform: [
        this.transform.m00(), this.transform.m01(), this.transform.m02(),
        this.transform.m10(), this.transform.m11(), this.transform.m12(),
        this.transform.m20(), this.transform.m21(), this.transform.m22()
      ],
      image: RenderImage.serializeRenderImageable( this.image ),
      extendX: this.extendX,
      extendY: this.extendY,
      resampleType: this.resampleType
    };
  }

  public static override deserialize( obj: SerializedRenderImage ): RenderImage {
    return new RenderImage(
      Matrix3.rowMajor(
        obj.transform[ 0 ], obj.transform[ 1 ], obj.transform[ 2 ],
        obj.transform[ 3 ], obj.transform[ 4 ], obj.transform[ 5 ],
        obj.transform[ 6 ], obj.transform[ 7 ], obj.transform[ 8 ]
      ),
      RenderImage.deserializeRenderImageable( obj.image ),
      obj.extendX,
      obj.extendY,
      obj.resampleType
    );
  }

  public static serializeRenderImageable( imageable: RenderImageable ): SerializedRenderImageable {
    return {
      width: imageable.width,
      height: imageable.height,
      isFullyOpaque: imageable.isFullyOpaque,
      data: _.range( 0, imageable.height ).flatMap( y => {
        return _.range( 0, imageable.width ).flatMap( x => {
          const color = imageable.evaluate( x, y );
          return [
            color.x,
            color.y,
            color.z,
            color.w
          ];
        } );
      } )
    };
  }

  public static deserializeRenderImageable( obj: SerializedRenderImageable ): RenderImageable {
    return {
      width: obj.width,
      height: obj.height,
      isFullyOpaque: obj.isFullyOpaque,
      evaluate: ( x: number, y: number ) => {
        const index = ( y * obj.width + x ) * 4;
        return new Vector4( obj.data[ index ], obj.data[ index + 1 ], obj.data[ index + 2 ], obj.data[ index + 3 ] );
      }
    };
  }
}

scenery.register( 'RenderImage', RenderImage );

export type SerializedRenderImage = {
  type: 'RenderImage';
  transform: number[];
  image: SerializedRenderImageable;
  extendX: RenderExtend;
  extendY: RenderExtend;
  resampleType: RenderResampleType;
};
