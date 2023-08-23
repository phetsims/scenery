// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, FaceConversion, getPolygonFilterExtraPixels, getPolygonFilterGridOffset, IntegerEdge, LineIntersector, LineSplitter, OutputRaster, PolygonFilterType, PolygonMitchellNetravali, RationalBoundary, RationalFace, RationalHalfEdge, RenderableFace, RenderColor, RenderPath, RenderPathProgram, RenderProgram, RenderProgramNeeds, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import IntentionalAny from '../../../../../phet-core/js/types/IntentionalAny.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { optionize3 } from '../../../../../phet-core/js/optionize.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export type RasterizationOptions = {
  outputRasterOffset?: Vector2;
  polygonFiltering?: PolygonFilterType;

  edgeIntersectionMethod?: 'quadratic' | 'boundsTree' | 'arrayBoundsTree';

  renderableFaceMethod?: 'polygonal' | 'edged' | 'fullyCombined' | 'simplifyingCombined' | 'traced';

  splitLinearGradients?: boolean;
  splitRadialGradients?: boolean;
};

const DEFAULT_OPTIONS = {
  outputRasterOffset: Vector2.ZERO,
  polygonFiltering: PolygonFilterType.Box,
  edgeIntersectionMethod: 'arrayBoundsTree',
  renderableFaceMethod: 'traced',
  splitLinearGradients: true,
  splitRadialGradients: true
} as const;

let debugData: Record<string, IntentionalAny> | null = null;

const scratchFullAreaVector = new Vector2( 0, 0 );

const nanVector = new Vector2( NaN, NaN );

class RasterizationContext {
  public constructor(
    public outputRaster: OutputRaster,
    public renderProgram: RenderProgram,
    public constColor: Vector4 | null,
    public constSRGBColor: Vector4 | null,
    public outputRasterOffset: Vector2,
    public bounds: Bounds2,
    public polygonFiltering: PolygonFilterType,
    public needs: RenderProgramNeeds
  ) {}
}

export default class Rasterize {

  private static getRenderProgrammedFaces( renderProgram: RenderProgram, faces: RationalFace[] ): RationalFace[] {
    const renderProgrammedFaces: RationalFace[] = [];

    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];

      face.postWindingRenderProgram( renderProgram );

      // Drop faces that will be fully transparent
      const isFullyTransparent = face.renderProgram instanceof RenderColor && face.renderProgram.color.w <= 1e-8;

      if ( !isFullyTransparent ) {
        renderProgrammedFaces.push( face );
      }
    }

    return renderProgrammedFaces;
  }

  // TODO: reconstitute this into an alternative non-binary setup! (filtering will be a bit more annoying?)
  // private static fullRasterize(
  //   outputRaster: OutputRaster,
  //   renderProgram: RenderProgram,
  //   clippableFace: ClippableFace,
  //   constColor: Vector4 | null,
  //   bounds: Bounds2, // TODO: check it's integral
  //   translation: Vector2
  // ): void {
  //   const pixelBounds = Bounds2.NOTHING.copy();
  //   const minX = bounds.minX;
  //   const minY = bounds.minY;
  //   const maxX = bounds.maxX;
  //   const maxY = bounds.maxY;
  //
  //   for ( let y = minY; y < maxY; y++ ) {
  //     pixelBounds.minY = y;
  //     pixelBounds.maxY = y + 1;
  //     for ( let x = minX; x < maxX; x++ ) {
  //       pixelBounds.minX = x;
  //       pixelBounds.maxX = x + 1;
  //
  //       const pixelFace = clippableFace.getClipped( pixelBounds );
  //       const area = pixelFace.getArea();
  //       if ( area > 1e-8 ) {
  //         Rasterize.addPartialPixel(
  //           outputRaster, renderProgram, constColor, translation,
  //           pixelFace, area, x, y
  //         );
  //       }
  //     }
  //   }
  // }

  private static addFilterPixel(
    context: RasterizationContext,
    pixelFace: ClippableFace | null,
    area: number,
    x: number,
    y: number,
    color: Vector4
  ): void {

    const polygonFiltering = context.polygonFiltering;
    const bounds = context.bounds;
    const outputRaster = context.outputRaster;
    const outputRasterOffset = context.outputRasterOffset;

    assert && assert( polygonFiltering === PolygonFilterType.Bilinear || polygonFiltering === PolygonFilterType.MitchellNetravali,
      'Only supports these filters currently' );

    const expand = polygonFiltering === PolygonFilterType.MitchellNetravali ? 2 : 1;

    // e.g. x - px is 0,-1 for bilinear, 1,-2 for mitchell-netravali
    // px = x + (0,1) for bilinear, x + (-1,2) for mitchell-netravali

    const subIterMinX = x - expand + 1;
    const subIterMinY = y - expand + 1;
    const subIterMaxX = x + expand + 1;
    const subIterMaxY = y + expand + 1;

    for ( let py = subIterMinY; py < subIterMaxY; py++ ) {

      const pixelY = py - 0.5;
      // TODO: put these in the subIter values above
      if ( pixelY < bounds.minY || pixelY >= bounds.maxY ) {
        continue;
      }

      for ( let px = subIterMinX; px < subIterMaxX; px++ ) {

        const pixelX = px - 0.5;
        if ( pixelX < bounds.minX || pixelX >= bounds.maxX ) {
          continue;
        }

        let contribution;

        // If it has a full pixel of area, we can simplify computation SIGNIFICANTLY
        if ( area > 1 - 1e-8 ) {
          // only bilinear and mitchell-netravali
          contribution = polygonFiltering === PolygonFilterType.MitchellNetravali ? PolygonMitchellNetravali.evaluateFull( px, py, x, y ) : 0.25;
        }
        else {
          assert && assert( pixelFace );

          if ( assertSlow ) {
            // TODO: implement these for polygonal faces
            const edges = pixelFace!.toEdgedFace().edges;

            assertSlow( edges.every( edge => {
              return edge.startPoint.x >= x && edge.startPoint.x <= x + 1 &&
                     edge.startPoint.y >= y && edge.startPoint.y <= y + 1 &&
                     edge.endPoint.x >= x && edge.endPoint.x <= x + 1 &&
                     edge.endPoint.y >= y && edge.endPoint.y <= y + 1;
            } ) );
          }

          contribution = polygonFiltering === PolygonFilterType.MitchellNetravali ?
                         pixelFace!.getMitchellNetravaliFiltered( px, py, x, y ) :
                         pixelFace!.getBilinearFiltered( px, py, x, y );
        }

        outputRaster.addPartialPixel( color.timesScalar( contribution ), pixelX + outputRasterOffset.x, pixelY + outputRasterOffset.y );
      }
    }
  }

  // TODO: inline eventually
  private static addPartialPixel(
    context: RasterizationContext,
    pixelFace: ClippableFace,
    area: number,
    x: number,
    y: number
  ): void {
    if ( assert ) {
      debugData!.areas.push( new Bounds2( x, y, x + 1, y + 1 ) );
    }

    // TODO: potentially cache the centroid, if we have multiple overlapping gradients?
    const color = context.constColor || context.renderProgram.evaluate(
      pixelFace,
      context.needs.needsArea ? area : NaN, // NaNs to hopefully hard-error
      context.needs.needsCentroid ? pixelFace.getCentroid( area ) : nanVector,
      x,
      y,
      x + 1,
      y + 1
    );

    if ( context.polygonFiltering === PolygonFilterType.Box ) {
      context.outputRaster.addPartialPixel( color.timesScalar( area ), x + context.outputRasterOffset.x, y + context.outputRasterOffset.y );
    }
    else {
      Rasterize.addFilterPixel(
        context,
        pixelFace, area, x, y, color
      );
    }
  }

  // TODO: inline eventually
  private static addFullArea(
    context: RasterizationContext,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {
    if ( assert ) {
      debugData!.areas.push( new Bounds2( minX, minY, maxX, maxY ) );
    }

    const constColor = context.constColor;

    if ( constColor ) {
      assert && assert( !context.needs.needsArea && !context.needs.needsCentroid );

      if ( context.polygonFiltering === PolygonFilterType.Box ) {
        if ( context.constSRGBColor ) {
          context.outputRaster.addFullRegionSRGB255(
            context.constSRGBColor,
            minX + context.outputRasterOffset.x,
            minY + context.outputRasterOffset.y,
            maxX - minX,
            maxY - minY
          );
        }
        else {
          context.outputRaster.addFullRegion(
            constColor,
            minX + context.outputRasterOffset.x,
            minY + context.outputRasterOffset.y,
            maxX - minX,
            maxY - minY
          );
        }
      }
      else {
        // TODO: ideally we can optimize this if it has a significant number of contained pixels. We only need to
        // "filter" the outside ones (the inside will be a constant color)
        for ( let y = minY; y < maxY; y++ ) {
          for ( let x = minX; x < maxX; x++ ) {
            Rasterize.addFilterPixel( context, null, 1, x, y, constColor );
          }
        }
      }
    }
    else {
      const renderProgram = context.renderProgram;
      const needs = context.needs;
      const polygonFiltering = context.polygonFiltering;
      const outputRaster = context.outputRaster;
      const outputRasterOffset = context.outputRasterOffset;

      const pixelArea = needs.needsArea ? 1 : NaN; // NaNs to hopefully hard-error
      for ( let y = minY; y < maxY; y++ ) {
        for ( let x = minX; x < maxX; x++ ) {
          const color = renderProgram.evaluate(
            null,
            pixelArea,
            needs.needsCentroid ? scratchFullAreaVector.setXY( x + 0.5, y + 0.5 ) : nanVector,
            x,
            y,
            x + 1,
            y + 1
          );
          if ( polygonFiltering === PolygonFilterType.Box ) {
            outputRaster.addFullPixel( color, x + outputRasterOffset.x, y + outputRasterOffset.y );
          }
          else {
            Rasterize.addFilterPixel( context, null, 1, x, y, color );
          }
        }
      }
    }
  }

  private static binaryInternalRasterize(
    context: RasterizationContext,
    clippableFace: ClippableFace,
    area: number,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {

    // TODO: more advanced handling

    const xDiff = maxX - minX;
    const yDiff = maxY - minY;

    assert && assert( xDiff >= 1 && yDiff >= 1 );
    assert && assert( Number.isInteger( xDiff ) && Number.isInteger( yDiff ) );
    assert && assert( context.polygonFiltering === PolygonFilterType.Box ? Number.isInteger( minX ) : minX - Math.floor( minX ) === 0.5 );
    assert && assert( context.polygonFiltering === PolygonFilterType.Box ? Number.isInteger( minY ) : minY - Math.floor( minY ) === 0.5 );
    assert && assert( context.polygonFiltering === PolygonFilterType.Box ? Number.isInteger( maxX ) : maxX - Math.floor( maxX ) === 0.5 );
    assert && assert( context.polygonFiltering === PolygonFilterType.Box ? Number.isInteger( maxY ) : maxY - Math.floor( maxY ) === 0.5 );

    if ( area > 1e-8 ) {
      if ( area >= ( maxX - minX ) * ( maxY - minY ) - 1e-8 ) {
        Rasterize.addFullArea(
          context,
          minX, minY, maxX, maxY
        );
      }
      else if ( xDiff === 1 && yDiff === 1 ) {
        Rasterize.addPartialPixel(
          context,
          clippableFace, area, minX, minY
        );
      }
      else {
        const averageX = ( minX + maxX ) / 2;
        const averageY = ( minY + maxY ) / 2;

        if ( xDiff > yDiff ) {
          const xSplit = minX + Math.floor( 0.5 * xDiff );

          assert && assert( xSplit !== minX && xSplit !== maxX );

          // TODO: If this is the LAST level of clipping, can we perhaps skip the actual face output (and only get
          // TODO: area output)?
          const { minFace, maxFace } = clippableFace.getBinaryXClip( xSplit, averageY );

          if ( assertSlow ) {
            const oldMinFace = clippableFace.getClipped( new Bounds2( minX, minY, xSplit, maxY ) );
            const oldMaxFace = clippableFace.getClipped( new Bounds2( xSplit, minY, maxX, maxY ) );

            if ( Math.abs( minFace.getArea() - oldMinFace.getArea() ) > 1e-8 || Math.abs( maxFace.getArea() - oldMaxFace.getArea() ) > 1e-8 ) {
              assertSlow( false, 'binary X clip issue' );
            }
          }

          const minArea = minFace.getArea();
          const maxArea = maxFace.getArea();

          if ( minArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              context,
              minFace, minArea, minX, minY, xSplit, maxY
            );
          }
          if ( maxArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              context,
              maxFace, maxArea, xSplit, minY, maxX, maxY
            );
          }
        }
        else {
          const ySplit = minY + Math.floor( 0.5 * yDiff );

          const { minFace, maxFace } = clippableFace.getBinaryYClip( ySplit, averageX );

          if ( assertSlow ) {
            const oldMinFace = clippableFace.getClipped( new Bounds2( minX, minY, maxX, ySplit ) );
            const oldMaxFace = clippableFace.getClipped( new Bounds2( minX, ySplit, maxX, maxY ) );

            if ( Math.abs( minFace.getArea() - oldMinFace.getArea() ) > 1e-8 || Math.abs( maxFace.getArea() - oldMaxFace.getArea() ) > 1e-8 ) {
              assertSlow( false, 'binary Y clip issue' );
            }
          }

          const minArea = minFace.getArea();
          const maxArea = maxFace.getArea();

          if ( minArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              context,
              minFace, minArea, minX, minY, maxX, ySplit
            );
          }
          if ( maxArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              context,
              maxFace, maxArea, minX, ySplit, maxX, maxY
            );
          }
        }
      }
    }
  }

  private static rasterizeAccumulate(
    outputRaster: OutputRaster,
    renderableFaces: RenderableFace[],
    bounds: Bounds2,
    gridBounds: Bounds2,
    outputRasterOffset: Vector2,
    polygonFiltering: PolygonFilterType
  ): void {
    for ( let i = 0; i < renderableFaces.length; i++ ) {
      const renderableFace = renderableFaces[ i ];
      const face = renderableFace.face;
      const renderProgram = renderableFace.renderProgram;
      const polygonalBounds = renderableFace.bounds;

      const faceDebugData: IntentionalAny = assert ? {
        face: face,
        pixels: [],
        areas: []
      } : null;
      if ( assert ) {
        debugData!.faceDebugData = debugData!.faceDebugData || [];
        debugData!.faceDebugData.push( faceDebugData );
      }

      const constColor = renderProgram instanceof RenderColor ? renderProgram.color : null;
      const constSRGBColor = constColor !== null ? RenderColor.convertLinearPremultipliedToSRGB( constColor ) : null;

      const context = new RasterizationContext(
        outputRaster,
        renderProgram,
        constColor,
        constSRGBColor,
        outputRasterOffset,
        bounds,
        polygonFiltering,
        renderProgram.getNeeds()
      );

      // For filtering, we'll want to round our faceBounds to the nearest (shifted) integer.
      const gridOffset = getPolygonFilterGridOffset( context.polygonFiltering );
      const faceBounds = polygonalBounds.intersection( gridBounds ).shiftedXY( gridOffset, gridOffset ).roundedOut().shiftedXY( -gridOffset, -gridOffset );

      // We will clip off anything outside the "bounds", since if we're based on EdgedFace we don't want those "fake"
      // edges that might be outside.
      const clippableFace = renderableFace.face.getClipped( faceBounds );
      if ( assert ) {
        faceDebugData.clippableFace = renderableFace.face;
        faceDebugData.clippedFace = clippableFace;
      }

      Rasterize.binaryInternalRasterize(
        context, clippableFace, clippableFace.getArea(), faceBounds.minX, faceBounds.minY, faceBounds.maxX, faceBounds.maxY
      );
    }
  }

  public static rasterize( renderProgram: RenderProgram, outputRaster: OutputRaster, bounds: Bounds2, providedOptions?: RasterizationOptions ): void {

    // Coordinate frames:
    //
    // First, we start with the RenderProgram coordinate frame (the coordinate frame of the paths inside the renderPrograms,
    // and the bounds provided to us.
    //
    // We will then transfer over to the "integer" coordinate frame, where each of our input edges will have
    // integer-valued coordinates. Additionally, we've shifted/scaled this, so that the integers lie within about
    // 20-bits of precision (and are centered in the integer grid). We do this so that we can do exact operations for
    // intersection/etc., which solves most of the robustness issues (the intersection point x,y between two of these
    // line segments can be stored each with a 64-bit numerator and a 64-bit denominator, and we can do the needed
    // arithmetic with the rationals).
    //
    // Once we have determined the intersections AND connected half-edges (which requires sorting the half-edges with
    // the exact values), we can then transfer back to the RenderProgram coordinate frame, and rasterize the faces
    // in this coordinate frame.
    //
    // Of note, when we're filtering with bilinear or Mitchell-Netravali filters, we'll be cutting up the faces into
    // half-pixel offset expanded regions, so that we can evaluate the filters AT the pixel centers.

    assert && assert( bounds.isValid() && !bounds.isEmpty(), 'Rasterization bounds should be valid and non-empty' );
    assert && assert( Number.isInteger( bounds.left ) && Number.isInteger( bounds.top ) && Number.isInteger( bounds.right ) && Number.isInteger( bounds.bottom ) );

    const options = optionize3<RasterizationOptions>()( {}, DEFAULT_OPTIONS, providedOptions );

    if ( assert ) {
      debugData = {
        areas: []
      };

      // NOTE: find a better way of doing this?
      // @ts-expect-error
      window.debugData = debugData;
    }

    const polygonFiltering: PolygonFilterType = options.polygonFiltering;

    const filterAdditionalPixels = getPolygonFilterExtraPixels( polygonFiltering );
    const filterGridOffset = getPolygonFilterGridOffset( polygonFiltering );

    const outputWidth = bounds.width;
    const outputHeight = bounds.height;
    const gridWidth = outputWidth + filterAdditionalPixels;
    const gridHeight = outputHeight + filterAdditionalPixels;

    // in RenderProgram coordinate frame
    const gridBounds = new Bounds2(
      bounds.minX + filterGridOffset,
      bounds.minY + filterGridOffset,
      bounds.maxX + filterGridOffset + filterAdditionalPixels,
      bounds.maxY + filterGridOffset + filterAdditionalPixels
    );

    // Keep us at 20 bits of precision (after rounding)
    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( Math.max( gridWidth, gridHeight ) ) ) );
    if ( assert ) { debugData!.scale = scale; }

    // -( scale * ( bounds.minX + filterGridOffset.x ) + translation.x ) = scale * ( bounds.maxX + filterGridOffset.x ) + translation.x
    const translation = new Vector2(
      -0.5 * scale * ( 2 * filterGridOffset + filterAdditionalPixels + bounds.minX + bounds.maxX ),
      -0.5 * scale * ( 2 * filterGridOffset + filterAdditionalPixels + bounds.minY + bounds.maxY )
    );
    if ( assert ) { debugData!.translation = translation; }

    const toIntegerMatrix = Matrix3.affine( scale, 0, translation.x, 0, scale, translation.y );
    if ( assert ) { debugData!.toIntegerMatrix = toIntegerMatrix; }

    const fromIntegerMatrix = toIntegerMatrix.inverted();
    if ( assert ) { debugData!.fromIntegerMatrix = fromIntegerMatrix; }

    // Verify our math! Make sure we will be perfectly centered in our integer grid!
    assert && assert( Math.abs( ( scale * gridBounds.minX + translation.x ) + ( scale * gridBounds.maxX + translation.x ) ) < 1e-10 );
    assert && assert( Math.abs( ( scale * gridBounds.minY + translation.y ) + ( scale * gridBounds.maxY + translation.y ) ) < 1e-10 );

    const integerBounds = gridBounds.transformed( toIntegerMatrix );
    if ( assert ) { debugData!.integerBounds = integerBounds; }

    const paths: RenderPath[] = [];
    renderProgram.depthFirst( program => {
      if ( program instanceof RenderPathProgram && program.path !== null ) {
        paths.push( program.path );
      }
    } );
    const backgroundPath = new RenderPath( 'nonzero', [
      [
        bounds.leftTop,
        bounds.rightTop,
        bounds.rightBottom,
        bounds.leftBottom
      ]
    ] );
    paths.push( backgroundPath );

    const integerEdges = IntegerEdge.clipScaleToIntegerEdges( paths, gridBounds, toIntegerMatrix );
    if ( assert ) { debugData!.integerEdges = integerEdges; }

    // TODO: optional hilbert space-fill sort here?

    if ( options.edgeIntersectionMethod === 'quadratic' ) {
      LineIntersector.edgeIntersectionQuadratic( integerEdges );
    }
    else if ( options.edgeIntersectionMethod === 'boundsTree' ) {
      LineIntersector.edgeIntersectionBoundsTree( integerEdges );
    }
    else if ( options.edgeIntersectionMethod === 'arrayBoundsTree' ) {
      LineIntersector.edgeIntersectionArrayBoundsTree( integerEdges );
    }
    else {
      throw new Error( `unknown edgeIntersectionMethod: ${options.edgeIntersectionMethod}` );
    }

    const rationalHalfEdges = LineSplitter.splitIntegerEdges( integerEdges );

    rationalHalfEdges.sort( ( a, b ) => a.compare( b ) );

    const filteredRationalHalfEdges = RationalHalfEdge.filterAndConnectHalfEdges( rationalHalfEdges );
    if ( assert ) { debugData!.filteredRationalHalfEdges = filteredRationalHalfEdges; }

    const innerBoundaries: RationalBoundary[] = [];
    const outerBoundaries: RationalBoundary[] = [];
    const faces: RationalFace[] = [];
    if ( assert ) {
      debugData!.innerBoundaries = innerBoundaries;
      debugData!.outerBoundaries = outerBoundaries;
      debugData!.faces = faces;
    }
    RationalFace.traceBoundaries( filteredRationalHalfEdges, innerBoundaries, outerBoundaries, faces );

    let faceHoleLog = null;
    if ( assert ) {
      debugData!.boundaryDebugData = [];
      faceHoleLog = { entries: debugData!.boundaryDebugData };
    }

    const exteriorBoundary = RationalFace.computeFaceHoles(
      integerBounds,
      outerBoundaries,
      faces,
      faceHoleLog
    );

    // For ease of use, an unbounded face (it is essentially fake)
    const unboundedFace = RationalFace.createUnboundedFace( exteriorBoundary );
    if ( assert ) {
      debugData!.unboundedFace = unboundedFace;
    }

    RationalFace.computeWindingMaps( filteredRationalHalfEdges, unboundedFace );

    const renderedFaces = Rasterize.getRenderProgrammedFaces( renderProgram, faces );

    let renderableFaces: RenderableFace[];
    if ( options.renderableFaceMethod === 'polygonal' ) {
      renderableFaces = FaceConversion.toPolygonalRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'edged' ) {
      renderableFaces = FaceConversion.toEdgedRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'fullyCombined' ) {
      renderableFaces = FaceConversion.toFullyCombinedRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'simplifyingCombined' ) {
      renderableFaces = FaceConversion.toSimplifyingCombinedRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'traced' ) {
      renderableFaces = FaceConversion.toTracedRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else {
      throw new Error( 'unknown renderableFaceMethod' );
    }

    if ( options.splitLinearGradients ) {
      renderableFaces = renderableFaces.flatMap( face => face.splitLinearGradients() );
    }
    if ( options.splitRadialGradients ) {
      renderableFaces = renderableFaces.flatMap( face => face.splitRadialGradients() );
    }

    Rasterize.rasterizeAccumulate(
      outputRaster,
      renderableFaces,
      bounds,
      gridBounds,
      options.outputRasterOffset,
      polygonFiltering
    );
  }

  public static imageDataToCanvas( imageData: ImageData ): HTMLCanvasElement {
    const canvas = document.createElement( 'canvas' );
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const context = canvas.getContext( '2d' )!;
    context.putImageData( imageData, 0, 0 );
    return canvas;
  }
}

scenery.register( 'Rasterize', Rasterize );
