// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BoundedSubpath, ClippableFace, FaceConversion, getPolygonFilterGridBounds, getPolygonFilterGridOffset, getPolygonFilterWidth, HilbertMapping, IntegerEdge, LineIntersector, LineSplitter, OutputRaster, PolygonFilterType, PolygonMitchellNetravali, RasterLog, RasterTileLog, RationalBoundary, RationalFace, RationalHalfEdge, RenderableFace, RenderColor, RenderPath, RenderPathBoolean, RenderPathReplacer, RenderProgram, RenderProgramNeeds, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { optionize3 } from '../../../../../phet-core/js/optionize.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export type RasterizationOptions = {
  // TODO: doc
  outputRasterOffset?: Vector2;

  tileSize?: number;

  // TODO: doc
  polygonFiltering?: PolygonFilterType;

  // We'll expand the filter window by this multiplier. If it is not 1, it will potentially drop performance
  // significantly (we won't be able to grid-clip to do it efficiently, and it might cover significantly more area).
  polygonFilterWindowMultiplier?: number;

  // TODO: consistent naming conventions
  edgeIntersectionSortMethod?: 'none' | 'center-size' | 'min-max' | 'min-max-size' | 'center-min-max' | 'random';

  edgeIntersectionMethod?: 'quadratic' | 'boundsTree' | 'arrayBoundsTree';

  renderableFaceMethod?: 'polygonal' | 'edged' | 'fullyCombined' | 'simplifyingCombined' | 'traced';

  splitPrograms?: boolean;

  log?: RasterLog | null;
};

const DEFAULT_OPTIONS = {
  outputRasterOffset: Vector2.ZERO,
  tileSize: Number.POSITIVE_INFINITY,
  polygonFiltering: PolygonFilterType.Box,
  polygonFilterWindowMultiplier: 1,
  edgeIntersectionSortMethod: 'center-min-max',
  edgeIntersectionMethod: 'arrayBoundsTree',
  renderableFaceMethod: 'traced',
  splitPrograms: true,
  log: null
} as const;

const scratchFullAreaVector = new Vector2( 0, 0 );

const nanVector = new Vector2( NaN, NaN );

class RasterizationContext {
  public constructor(
    public outputRaster: OutputRaster,
    public renderProgram: RenderProgram,
    public constClientColor: Vector4 | null,
    public constOutputColor: Vector4 | null,
    public outputRasterOffset: Vector2,
    public bounds: Bounds2,
    public polygonFiltering: PolygonFilterType,
    public polygonFilterWindowMultiplier: number,
    public needs: RenderProgramNeeds,
    public log: RasterLog | null
  ) {}
}

export default class Rasterize {

  private static getRenderProgrammedFaces( renderProgram: RenderProgram, faces: RationalFace[] ): RationalFace[] {
    const renderProgrammedFaces: RationalFace[] = [];

    const replacer = new RenderPathReplacer( renderProgram.simplified() );

    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];

      face.renderProgram = replacer.replace( face.getIncludedRenderPaths() );

      if ( assertSlow ) {
        const inclusionSet = face.getIncludedRenderPaths();

        const checkProgram = renderProgram.withPathInclusion( renderPath => inclusionSet.has( renderPath ) ).simplified();

        assertSlow( face.renderProgram.equals( checkProgram ), 'Replacer/simplifier error' );
      }

      // Drop faces that will be fully transparent
      const isFullyTransparent = face.renderProgram instanceof RenderColor && face.renderProgram.color.w <= 1e-8;

      if ( !isFullyTransparent ) {
        renderProgrammedFaces.push( face );
      }
    }

    return renderProgrammedFaces;
  }

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

        outputRaster.addClientPartialPixel( color.timesScalar( contribution ), pixelX + outputRasterOffset.x, pixelY + outputRasterOffset.y );
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
    // TODO: potentially cache the centroid, if we have multiple overlapping gradients?
    const color = context.constClientColor || context.renderProgram.evaluate(
      pixelFace,
      context.needs.needsArea ? area : NaN, // NaNs to hopefully hard-error
      context.needs.needsCentroid ? pixelFace.getCentroid( area ) : nanVector,
      x,
      y,
      x + 1,
      y + 1
    );

    if ( context.polygonFiltering === PolygonFilterType.Box ) {
      context.outputRaster.addClientPartialPixel( color.timesScalar( area ), x + context.outputRasterOffset.x, y + context.outputRasterOffset.y );
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
    const constClientColor = context.constClientColor;

    if ( constClientColor ) {
      assert && assert( !context.needs.needsArea && !context.needs.needsCentroid );

      if ( context.polygonFiltering === PolygonFilterType.Box ) {
        if ( context.constOutputColor ) {
          context.outputRaster.addOutputFullRegion(
            context.constOutputColor,
            minX + context.outputRasterOffset.x,
            minY + context.outputRasterOffset.y,
            maxX - minX,
            maxY - minY
          );
        }
        else {
          context.outputRaster.addClientFullRegion(
            constClientColor,
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
            Rasterize.addFilterPixel( context, null, 1, x, y, constClientColor );
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
            outputRaster.addClientFullPixel( color, x + outputRasterOffset.x, y + outputRasterOffset.y );
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
        if ( context.log ) { context.log.fullAreas.push( new Bounds2( minX, minY, maxX, maxY ) ); }
        Rasterize.addFullArea(
          context,
          minX, minY, maxX, maxY
        );
      }
      else if ( xDiff === 1 && yDiff === 1 ) {
        if ( context.log ) { context.log.partialAreas.push( new Bounds2( minX, minY, maxX, maxY ) ); }
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

  private static windowedFilterRasterize(
    context: RasterizationContext,
    clippableFace: ClippableFace,
    faceBounds: Bounds2
  ): void {
    // USE getPolygonFilterGridBounds

    const outputMinX = context.bounds.minX;
    const outputMaxX = context.bounds.maxX;
    const outputMinY = context.bounds.minY;
    const outputMaxY = context.bounds.maxY;

    const filterWidth = getPolygonFilterWidth( context.polygonFiltering ) * context.polygonFilterWindowMultiplier;
    const filterArea = filterWidth * filterWidth;
    const filterExtension = 0.5 * ( filterWidth - 1 );
    const filterMin = -filterExtension;
    const filterMax = 1 + filterExtension;
    const descaleMatrix = Matrix3.scaling( 1 / context.polygonFilterWindowMultiplier );

    const quadClip = ( face: ClippableFace, x: number, y: number ): {
      minXMinYFace: ClippableFace;
      minXMaxYFace: ClippableFace;
      maxXMinYFace: ClippableFace;
      maxXMaxYFace: ClippableFace;
    } => {
      const xClipped = face.getBinaryXClip( x, y );
      const minXFace = xClipped.minFace;
      const maxXFace = xClipped.maxFace;

      const minXYClipped = minXFace.getBinaryYClip( y, x );
      const maxXYClipped = maxXFace.getBinaryYClip( y, x );
      return {
        minXMinYFace: minXYClipped.minFace,
        minXMaxYFace: minXYClipped.maxFace,
        maxXMinYFace: maxXYClipped.minFace,
        maxXMaxYFace: maxXYClipped.maxFace
      };
    };

    for ( let y = outputMinY; y < outputMaxY; y++ ) {

      const minY = y + filterMin;
      const maxY = y + filterMax;

      // If our filter window is outside of the face bounds, we can skip it entirely
      if ( minY >= faceBounds.maxY || maxY <= faceBounds.minY ) {
        continue;
      }

      for ( let x = outputMinX; x < outputMaxX; x++ ) {

        const minX = x + filterMin;
        const maxX = x + filterMax;

        // If our filter window is outside of the face bounds, we can skip it entirely
        if ( minX >= faceBounds.maxX || maxX <= faceBounds.minX ) {
          continue;
        }

        const clippedFace = clippableFace.getClipped( new Bounds2( minX, minY, maxX, maxY ) );
        const area = clippedFace.getArea();

        if ( area > 1e-8 ) {
          if ( context.constClientColor ) {
            const color = context.constClientColor;

            let contribution = 0;
            if ( context.polygonFiltering === PolygonFilterType.Box ) {
              contribution = area / filterArea;
            }
            else {
              // TODO: we don't have to transform the translation, could handle in the getBilinearFiltered() call etc.
              // get the face with our "pixel" center at the origin, and scaled to the filter window
              const unitFace = clippedFace.getTransformed( descaleMatrix.timesMatrix( Matrix3.translation( -( x + 0.5 ), -( y + 0.5 ) ) ) );

              const quadClipped = quadClip( unitFace, 0, 0 );

              if ( context.polygonFiltering === PolygonFilterType.Bilinear ) {
                contribution = quadClipped.minXMinYFace.getBilinearFiltered( 0, 0, -1, -1 ) +
                               quadClipped.minXMaxYFace.getBilinearFiltered( 0, 0, -1, 0 ) +
                               quadClipped.maxXMinYFace.getBilinearFiltered( 0, 0, 0, -1 ) +
                               quadClipped.maxXMaxYFace.getBilinearFiltered( 0, 0, 0, 0 );
              }
              else if ( context.polygonFiltering === PolygonFilterType.MitchellNetravali ) {
                const minMinQuad = quadClip( quadClipped.minXMinYFace, -1, -1 );
                const minMaxQuad = quadClip( quadClipped.minXMaxYFace, -1, 1 );
                const maxMinQuad = quadClip( quadClipped.maxXMinYFace, 1, -1 );
                const maxMaxQuad = quadClip( quadClipped.maxXMaxYFace, 1, 1 );

                contribution = minMinQuad.minXMinYFace.getMitchellNetravaliFiltered( 0, 0, -2, -2 ) +
                               minMinQuad.minXMaxYFace.getMitchellNetravaliFiltered( 0, 0, -2, -1 ) +
                               minMinQuad.maxXMinYFace.getMitchellNetravaliFiltered( 0, 0, -1, -2 ) +
                               minMinQuad.maxXMaxYFace.getMitchellNetravaliFiltered( 0, 0, -1, -1 ) +
                               minMaxQuad.minXMinYFace.getMitchellNetravaliFiltered( 0, 0, -2, 0 ) +
                               minMaxQuad.minXMaxYFace.getMitchellNetravaliFiltered( 0, 0, -2, 1 ) +
                               minMaxQuad.maxXMinYFace.getMitchellNetravaliFiltered( 0, 0, -1, 0 ) +
                               minMaxQuad.maxXMaxYFace.getMitchellNetravaliFiltered( 0, 0, -1, 1 ) +
                               maxMinQuad.minXMinYFace.getMitchellNetravaliFiltered( 0, 0, 0, -2 ) +
                               maxMinQuad.minXMaxYFace.getMitchellNetravaliFiltered( 0, 0, 0, -1 ) +
                               maxMinQuad.maxXMinYFace.getMitchellNetravaliFiltered( 0, 0, 1, -2 ) +
                               maxMinQuad.maxXMaxYFace.getMitchellNetravaliFiltered( 0, 0, 1, -1 ) +
                               maxMaxQuad.minXMinYFace.getMitchellNetravaliFiltered( 0, 0, 0, 0 ) +
                               maxMaxQuad.minXMaxYFace.getMitchellNetravaliFiltered( 0, 0, 0, 1 ) +
                               maxMaxQuad.maxXMinYFace.getMitchellNetravaliFiltered( 0, 0, 1, 0 ) +
                               maxMaxQuad.maxXMaxYFace.getMitchellNetravaliFiltered( 0, 0, 1, 1 );
              }
            }

            context.outputRaster.addClientPartialPixel( color.timesScalar( contribution ), x + context.outputRasterOffset.x, y + context.outputRasterOffset.y );
          }
          else {
            const transformMatrix = descaleMatrix.timesMatrix( Matrix3.translation( -( x + 0.5 ), -( y + 0.5 ) ) );
            const inverseMatrix = transformMatrix.inverted();
            const unitFace = clippedFace.getTransformed( transformMatrix );

            const splitAndProcessFace = ( ux: number, uy: number, face: ClippableFace, getContribution: ( f: ClippableFace ) => number ) => {
              const area = face.getArea();

              // TODO: performance is killed here, hopefully we don't really need it
              const minVector = inverseMatrix.timesVector2( new Vector2( ux, uy ) );
              const maxVector = inverseMatrix.timesVector2( new Vector2( ux + 1, uy + 1 ) );

              if ( area > 1e-8 ) {
                // TODO: SPLIT THIS
                const transformedArea = area * filterArea;
                const transformedFace = face.getTransformed( inverseMatrix );
                const transformedMinX = minVector.x;
                const transformedMinY = minVector.y;
                const transformedMaxX = maxVector.x;
                const transformedMaxY = maxVector.y;

                const binary = ( tFace: ClippableFace, tArea: number, tMinX: number, tMinY: number, tMaxX: number, tMaxY: number ) => {
                  const averageX = 0.5 * ( tMinX + tMaxX );
                  const averageY = 0.5 * ( tMinY + tMaxY );
                  if ( tMaxX - tMinX > 1 ) {
                    const { minFace, maxFace } = tFace.getBinaryXClip( averageX, averageY );
                    const minArea = minFace.getArea();
                    const maxArea = maxFace.getArea();

                    if ( minArea > 1e-8 ) {
                      binary( minFace, minArea, tMinX, tMinY, averageX, tMaxY );
                    }
                    if ( maxArea > 1e-8 ) {
                      binary( maxFace, maxArea, averageX, tMinY, tMaxX, tMaxY );
                    }
                  }
                  else if ( tMaxY - tMinY > 1 ) {
                    const { minFace, maxFace } = tFace.getBinaryYClip( averageY, averageX );
                    const minArea = minFace.getArea();
                    const maxArea = maxFace.getArea();

                    if ( minArea > 1e-8 ) {
                      binary( minFace, minArea, tMinX, tMinY, tMaxX, averageY );
                    }
                    if ( maxArea > 1e-8 ) {
                      binary( maxFace, maxArea, tMinX, averageY, tMaxX, tMaxY );
                    }
                  }
                  else {
                    const color = context.renderProgram.evaluate(
                      tFace,
                      context.needs.needsArea ? tArea : NaN, // NaNs to hopefully hard-error
                      context.needs.needsCentroid ? tFace.getCentroid( tArea ) : nanVector,
                      tMinX, tMinY, tMaxX, tMaxY
                    );
                    context.outputRaster.addClientPartialPixel( color.timesScalar( getContribution( tFace.getTransformed( transformMatrix ) ) ), x + context.outputRasterOffset.x, y + context.outputRasterOffset.y );
                  }
                };
                binary( transformedFace, transformedArea, transformedMinX, transformedMinY, transformedMaxX, transformedMaxY );
              }
            };

            if ( context.polygonFiltering === PolygonFilterType.Box ) {
              splitAndProcessFace( -0.5, -0.5, unitFace, face => face.getArea() );
            }
            else if ( context.polygonFiltering === PolygonFilterType.Bilinear ) {
              const quadClipped = quadClip( unitFace, 0, 0 );
              splitAndProcessFace( -1, -1, quadClipped.minXMinYFace, f => f.getBilinearFiltered( 0, 0, -1, -1 ) );
              splitAndProcessFace( -1, 0, quadClipped.minXMaxYFace, f => f.getBilinearFiltered( 0, 0, -1, 0 ) );
              splitAndProcessFace( 0, -1, quadClipped.maxXMinYFace, f => f.getBilinearFiltered( 0, 0, 0, -1 ) );
              splitAndProcessFace( 0, 0, quadClipped.maxXMaxYFace, f => f.getBilinearFiltered( 0, 0, 0, 0 ) );
            }
            else if ( context.polygonFiltering === PolygonFilterType.MitchellNetravali ) {
              const quadClipped = quadClip( unitFace, 0, 0 );

              const minMinQuad = quadClip( quadClipped.minXMinYFace, -1, -1 );
              const minMaxQuad = quadClip( quadClipped.minXMaxYFace, -1, 1 );
              const maxMinQuad = quadClip( quadClipped.maxXMinYFace, 1, -1 );
              const maxMaxQuad = quadClip( quadClipped.maxXMaxYFace, 1, 1 );

              // TODO: can factor out constants?
              splitAndProcessFace( -2, -2, minMinQuad.minXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -2, -2 ) );
              splitAndProcessFace( -2, -1, minMinQuad.minXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -2, -1 ) );
              splitAndProcessFace( -1, -2, minMinQuad.maxXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -1, -2 ) );
              splitAndProcessFace( -1, -1, minMinQuad.maxXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -1, -1 ) );
              splitAndProcessFace( -2, 0, minMaxQuad.minXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -2, 0 ) );
              splitAndProcessFace( -2, 1, minMaxQuad.minXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -2, 1 ) );
              splitAndProcessFace( -1, 0, minMaxQuad.maxXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -1, 0 ) );
              splitAndProcessFace( -1, 1, minMaxQuad.maxXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, -1, 1 ) );
              splitAndProcessFace( 0, -2, maxMinQuad.minXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 0, -2 ) );
              splitAndProcessFace( 0, -1, maxMinQuad.minXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 0, -1 ) );
              splitAndProcessFace( 1, -2, maxMinQuad.maxXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 1, -2 ) );
              splitAndProcessFace( 1, -1, maxMinQuad.maxXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 1, -1 ) );
              splitAndProcessFace( 0, 0, maxMaxQuad.minXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 0, 0 ) );
              splitAndProcessFace( 0, 1, maxMaxQuad.minXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 0, 1 ) );
              splitAndProcessFace( 1, 0, maxMaxQuad.maxXMinYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 1, 0 ) );
              splitAndProcessFace( 1, 1, maxMaxQuad.maxXMaxYFace, f => f.getMitchellNetravaliFiltered( 0, 0, 1, 1 ) );
            }
          }
        }
      }
    }
  }

  public static rasterizeAccumulate(
    outputRaster: OutputRaster,
    renderableFaces: RenderableFace[],
    bounds: Bounds2,
    contributionBounds: Bounds2,
    outputRasterOffset: Vector2,
    polygonFiltering: PolygonFilterType,
    polygonFilterWindowMultiplier: number,
    log: RasterLog | null
  ): void {
    for ( let i = 0; i < renderableFaces.length; i++ ) {
      const renderableFace = renderableFaces[ i ];
      const renderProgram = renderableFace.renderProgram;
      const polygonalBounds = renderableFace.bounds;

      // TODO: be really careful about the colorConverter... the copy() missing already hit me once.
      const constClientColor = renderProgram instanceof RenderColor ? renderProgram.color : null;
      const constOutputColor = constClientColor !== null ? outputRaster.colorConverter.clientToOutput( constClientColor ).copy() : null;

      const context = new RasterizationContext(
        outputRaster,
        renderProgram,
        constClientColor,
        constOutputColor,
        outputRasterOffset,
        bounds,
        polygonFiltering,
        polygonFilterWindowMultiplier,
        renderProgram.getNeeds(),
        log
      );

      if ( polygonFilterWindowMultiplier !== 1 ) {
        Rasterize.windowedFilterRasterize( context, renderableFace.face.getClipped( contributionBounds ), polygonalBounds.intersection( contributionBounds ) );
      }
      else {
        // For filtering, we'll want to round our faceBounds to the nearest (shifted) integer.
        const gridOffset = getPolygonFilterGridOffset( context.polygonFiltering );
        const faceBounds = polygonalBounds.intersection( contributionBounds ).shiftedXY( gridOffset, gridOffset ).roundedOut().shiftedXY( -gridOffset, -gridOffset );

        // We will clip off anything outside the "bounds", since if we're based on EdgedFace we don't want those "fake"
        // edges that might be outside.
        const clippableFace = renderableFace.face.getClipped( faceBounds );

        Rasterize.binaryInternalRasterize(
          context, clippableFace, clippableFace.getArea(), faceBounds.minX, faceBounds.minY, faceBounds.maxX, faceBounds.maxY
        );
      }
    }
  }

  public static rasterize(
    renderProgram: RenderProgram,
    outputRaster: OutputRaster,
    bounds: Bounds2,
    providedOptions?: RasterizationOptions
  ): void {

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

    // Just simplify things off-the-bat, so we don't need as much computation
    renderProgram = renderProgram.simplified();

    const options = optionize3<RasterizationOptions>()( {}, DEFAULT_OPTIONS, providedOptions );

    const log = options.log;

    const markStart = ( name: string ) => {
      log && window.performance && window.performance.mark( `${name}-start` );
    };
    const markEnd = ( name: string ) => {
      log && window.performance && window.performance.mark( `${name}-end` );
      log && window.performance && window.performance.measure( name, `${name}-start`, `${name}-end` );
    };

    markStart( 'rasterize' );

    const polygonFiltering: PolygonFilterType = options.polygonFiltering;
    const polygonFilterWindowMultiplier = options.polygonFilterWindowMultiplier;

    const paths = new Set<RenderPath>();
    renderProgram.depthFirst( program => {
      // TODO: we can filter based on hasPathBoolean, so we can skip subtrees
      if ( program instanceof RenderPathBoolean ) {
        paths.add( program.path );
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
    paths.add( backgroundPath );

    // The potentially filter-expanded bounds of content that could potentially affect pixels within our `bounds`,
    // in the RenderProgram coordinate frame.
    const contributionBounds = getPolygonFilterGridBounds( bounds, polygonFiltering, polygonFilterWindowMultiplier );

    markStart( 'path-bounds' );
    const boundedSubpaths = BoundedSubpath.fromPathSet( paths );
    markEnd( 'path-bounds' );

    // Keep us at 20 bits of precision (after rounding)
    const tileSize = options.tileSize;
    const maxSize = Math.min( tileSize, Math.max( contributionBounds.width, contributionBounds.height ) );
    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( maxSize ) ) );
    if ( log ) { log.scale = scale; }

    const combinedRenderableFaces: RenderableFace[] = [];
    for ( let y = contributionBounds.minY; y < contributionBounds.maxY; y += tileSize ) {
      for ( let x = contributionBounds.minX; x < contributionBounds.maxX; x += tileSize ) {
        const tileLog = log ? new RasterTileLog() : null;
        if ( log && tileLog ) { log.tileLogs.push( tileLog ); }

        // A slice of our contributionBounds
        const tileBounds = new Bounds2(
          x,
          y,
          Math.min( x + tileSize, contributionBounds.maxX ),
          Math.min( y + tileSize, contributionBounds.maxY )
        );

        // -( scale * ( tileBounds.minX + filterGridOffset.x ) + translation.x ) = scale * ( tileBounds.maxX + filterGridOffset.x ) + translation.x
        const translation = new Vector2(
          -0.5 * scale * ( tileBounds.minX + tileBounds.maxX ),
          -0.5 * scale * ( tileBounds.minY + tileBounds.maxY )
        );
        if ( tileLog ) { tileLog.translation = translation; }

        const toIntegerMatrix = Matrix3.affine( scale, 0, translation.x, 0, scale, translation.y );
        if ( tileLog ) { tileLog.toIntegerMatrix = toIntegerMatrix; }

        const fromIntegerMatrix = toIntegerMatrix.inverted();
        if ( tileLog ) { tileLog.fromIntegerMatrix = fromIntegerMatrix; }

        // Verify our math! Make sure we will be perfectly centered in our integer grid!
        assert && assert( Math.abs( ( scale * tileBounds.minX + translation.x ) + ( scale * tileBounds.maxX + translation.x ) ) < 1e-10 );
        assert && assert( Math.abs( ( scale * tileBounds.minY + translation.y ) + ( scale * tileBounds.maxY + translation.y ) ) < 1e-10 );

        markStart( 'clip-integer' );
        const integerEdges = IntegerEdge.clipScaleToIntegerEdges( boundedSubpaths, tileBounds, toIntegerMatrix );
        markEnd( 'clip-integer' );
        if ( tileLog ) { tileLog.integerEdges = integerEdges; }

        markStart( 'integer-sort' );
        // NOTE: Can also be 'none', we'll no-op
        if ( options.edgeIntersectionSortMethod === 'center-size' ) {
          HilbertMapping.sortCenterSize( integerEdges, 1 / ( scale * maxSize ) );
        }
        else if ( options.edgeIntersectionSortMethod === 'min-max' ) {
          HilbertMapping.sortMinMax( integerEdges, 1 / ( scale * maxSize ) );
        }
        else if ( options.edgeIntersectionSortMethod === 'min-max-size' ) {
          HilbertMapping.sortMinMaxSize( integerEdges, 1 / ( scale * maxSize ) );
        }
        else if ( options.edgeIntersectionSortMethod === 'center-min-max' ) {
          HilbertMapping.sortCenterMinMax( integerEdges, 1 / ( scale * maxSize ) );
        }
        else if ( options.edgeIntersectionSortMethod === 'random' ) {
          // NOTE: This is NOT designed for performance (it's for testing)
          // eslint-disable-next-line bad-sim-text
          const shuffled = _.shuffle( integerEdges );
          integerEdges.length = 0;
          integerEdges.push( ...shuffled );
        }
        markEnd( 'integer-sort' );

        markStart( 'integer-intersect' );
        if ( options.edgeIntersectionMethod === 'quadratic' ) {
          LineIntersector.edgeIntersectionQuadratic( integerEdges, tileLog );
        }
        else if ( options.edgeIntersectionMethod === 'boundsTree' ) {
          LineIntersector.edgeIntersectionBoundsTree( integerEdges, tileLog );
        }
        else if ( options.edgeIntersectionMethod === 'arrayBoundsTree' ) {
          LineIntersector.edgeIntersectionArrayBoundsTree( integerEdges, tileLog );
        }
        else {
          throw new Error( `unknown edgeIntersectionMethod: ${options.edgeIntersectionMethod}` );
        }
        markEnd( 'integer-intersect' );

        markStart( 'integer-split' );
        const rationalHalfEdges = LineSplitter.splitIntegerEdges( integerEdges );
        markEnd( 'integer-split' );

        markStart( 'edge-sort' );
        rationalHalfEdges.sort( ( a, b ) => a.compare( b ) );
        markEnd( 'edge-sort' );

        markStart( 'filter-connect' );
        let filteredRationalHalfEdges = RationalHalfEdge.filterAndConnectHalfEdges( rationalHalfEdges );
        markEnd( 'filter-connect' );
        if ( tileLog ) { tileLog.filteredRationalHalfEdges = filteredRationalHalfEdges; }

        const innerBoundaries: RationalBoundary[] = [];
        const outerBoundaries: RationalBoundary[] = [];
        const faces: RationalFace[] = [];
        if ( tileLog ) {
          tileLog.innerBoundaries = innerBoundaries;
          tileLog.outerBoundaries = outerBoundaries;
          tileLog.faces = faces;
        }
        markStart( 'trace-boundaries' );
        filteredRationalHalfEdges = RationalFace.traceBoundaries( filteredRationalHalfEdges, innerBoundaries, outerBoundaries, faces );
        markEnd( 'trace-boundaries' );
        if ( tileLog ) { tileLog.refilteredRationalHalfEdges = filteredRationalHalfEdges; }

        markStart( 'face-holes' );
        const exteriorBoundaries = RationalFace.computeFaceHolesWithOrderedWindingNumbers(
          outerBoundaries,
          faces
        );
        markEnd( 'face-holes' );
        assert && assert( exteriorBoundaries.length === 1, 'Should only have one external boundary, due to background' );
        const exteriorBoundary = exteriorBoundaries[ 0 ];

        // For ease of use, an unbounded face (it is essentially fake)
        const unboundedFace = RationalFace.createUnboundedFace( exteriorBoundary );
        if ( tileLog ) { tileLog.unboundedFace = unboundedFace; }

        markStart( 'winding-maps' );
        RationalFace.computeWindingMaps( filteredRationalHalfEdges, unboundedFace );
        markEnd( 'winding-maps' );

        markStart( 'render-programs' );
        const renderedFaces = Rasterize.getRenderProgrammedFaces( renderProgram, faces );
        if ( tileLog ) { tileLog.renderedFaces = renderedFaces; }
        markEnd( 'render-programs' );

        markStart( 'renderable-faces' );
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
        markEnd( 'renderable-faces' );
        if ( tileLog ) { tileLog.initialRenderableFaces = renderableFaces; }

        if ( options.splitPrograms ) {
          markStart( 'split-programs' );
          renderableFaces = renderableFaces.flatMap( face => face.split() );
          markEnd( 'split-programs' );
        }
        if ( tileLog ) { tileLog.renderableFaces = renderableFaces; }

        // TODO: If we had a RenderDepthSort, do a face combination here?

        combinedRenderableFaces.push( ...renderableFaces );
      }
    }

    if ( log ) { log.renderableFaces = combinedRenderableFaces; }

    markStart( 'rasterize-accumulate' );
    Rasterize.rasterizeAccumulate(
      outputRaster,
      combinedRenderableFaces,
      bounds,
      contributionBounds,
      options.outputRasterOffset,
      polygonFiltering,
      polygonFilterWindowMultiplier,
      log
    );
    markEnd( 'rasterize-accumulate' );

    markEnd( 'rasterize' );
  }

  public static imageDataToCanvas( imageData: ImageData ): HTMLCanvasElement {
    const canvas = document.createElement( 'canvas' );
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const context = ( imageData.colorSpace && imageData.colorSpace !== 'srgb' ) ?
                    canvas.getContext( '2d', { colorSpace: imageData.colorSpace } )! :
                    canvas.getContext( '2d' )!;
    context.putImageData( imageData, 0, 0 );
    return canvas;
  }

  public static writeImageDataToCanvas( imageData: ImageData, canvas: HTMLCanvasElement, context: CanvasRenderingContext2D ): void {
    if ( canvas.width !== imageData.width ) {
      canvas.width = imageData.width;
    }
    if ( canvas.height !== imageData.height ) {
      canvas.height = imageData.height;
    }
    context.putImageData( imageData, 0, 0 );
  }
}

scenery.register( 'Rasterize', Rasterize );
