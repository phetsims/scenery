// Copyright 2023, University of Colorado Boulder

/**
 * Similar to an DOM Canvas, but stores a vector representation of the relevant drawing commands.
 *
 * TODO: We really would want to "cache" all of the commands at once, build up a "pending" RenderProgram, and then
 * TODO: handle all the paths at once.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { CombinedRaster, CombinedRasterOptions, getPolygonFilterGridBounds, LinearEdge, PolygonalBoolean, PolygonalFace, PolygonClipping, PolygonFilterType, Rasterize, RenderableFace, RenderColor, RenderColorSpace, RenderExtend, RenderGradientStop, RenderLinearGradient, RenderLinearGradientAccuracy, RenderPath, RenderProgram, RenderRadialGradient, RenderRadialGradientAccuracy, RenderStack, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { combineOptions } from '../../../../../phet-core/js/optionize.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export default class VectorCanvas {

  public renderableFaces: RenderableFace[] = [];
  public bounds: Bounds2 = Bounds2.NOTHING;
  public gridBounds: Bounds2 = Bounds2.NOTHING;

  public constructor(
    public width: number,
    public height: number,
    public readonly colorSpace: 'srgb' | 'display-p3' = 'srgb',
    // TODO: proper options object?
    public readonly polygonFiltering: PolygonFilterType = PolygonFilterType.Box
  ) {
    this.updateWidthHeight( width, height );
  }

  // Assumes sRGB? We'll do srgb-linear blending to mimic for now?
  public fillColor( renderPath: RenderPath, color: Vector4 ): void {
    this.fillRenderProgram( renderPath, this.colorToRenderProgram( color ) );
  }

  public fillLinearGradient( renderPath: RenderPath, start: Vector2, end: Vector2, stops: { ratio: number; color: Vector4 }[], extend: RenderExtend, accuracy: RenderLinearGradientAccuracy = RenderLinearGradientAccuracy.SplitAccurate ): void {
    this.fillRenderProgram( renderPath, new RenderLinearGradient(
      Matrix3.IDENTITY, start, end, stops.map( stop => new RenderGradientStop( stop.ratio, this.colorToRenderProgram( stop.color ) ) ), extend, accuracy
    ) );

    this.renderableFaces = this.renderableFaces.flatMap( face => face.splitLinearGradients() );
  }

  public fillRadialGradient( renderPath: RenderPath, transform: Matrix3, start: Vector2, startRadius: number, end: Vector2, endRadius: number, stops: { ratio: number; color: Vector4 }[], extend: RenderExtend, accuracy: RenderRadialGradientAccuracy = RenderRadialGradientAccuracy.SplitAccurate ): void {
    this.fillRenderProgram( renderPath, new RenderRadialGradient(
      Matrix3.IDENTITY, start, startRadius, end, endRadius, stops.map( stop => new RenderGradientStop( stop.ratio, this.colorToRenderProgram( stop.color ) ) ), extend, accuracy
    ) );

    this.renderableFaces = this.renderableFaces.flatMap( face => face.splitRadialGradients() );
  }

  private colorToRenderProgram( color: Vector4 ): RenderProgram {
    return new RenderColor( color ).colorConverted( RenderColorSpace.sRGB, this.colorSpace === 'srgb' ? RenderColorSpace.premultipliedSRGB : RenderColorSpace.premultipliedDisplayP3 );
  }

  private fillRenderProgram( renderPath: RenderPath, renderProgram: RenderProgram ): void {
    const bounds = renderPath.getBounds();

    renderPath = new RenderPath( renderPath.fillRule, renderPath.subpaths.map( subpath => PolygonClipping.boundsClipPolygon( subpath, this.bounds ) ) );

    const newRenderableFaces: RenderableFace[] = [];

    for ( let i = 0; i < this.renderableFaces.length; i++ ) {
      const renderableFace = this.renderableFaces[ i ];

      if ( renderableFace.bounds.intersectsBounds( bounds ) ) {
        const existingRenderPath = new RenderPath( 'nonzero', renderableFace.face.toPolygonalFace().polygons );
        const overlaps = PolygonalBoolean.getOverlaps( existingRenderPath, renderPath );

        if ( overlaps.intersection.length ) {
          if ( overlaps.aOnly.length ) {
            const aOnlyFace = new PolygonalFace( overlaps.aOnly );
            newRenderableFaces.push( new RenderableFace(
              aOnlyFace,
              renderableFace.renderProgram,
              aOnlyFace.getBounds()
            ) );
          }
          const intersectionFace = new PolygonalFace( overlaps.intersection );
          newRenderableFaces.push( new RenderableFace(
            intersectionFace,
            new RenderStack( [ renderableFace.renderProgram, renderProgram ] ).simplified(),
            intersectionFace.getBounds()
          ) );
        }
        else {
          newRenderableFaces.push( renderableFace );
        }
      }
      else {
        newRenderableFaces.push( renderableFace );
      }
    }

    this.renderableFaces = newRenderableFaces;

    this.combineFaces();
  }

  private combineFaces(): void {
    // TODO: ONLY split linear/radial gradients AFTER we have combined faces!!!!

    for ( let i = 0; i < this.renderableFaces.length; i++ ) {
      const renderableFace = this.renderableFaces[ i ];
      const renderProgram = renderableFace.renderProgram;

      const compatibleFaces = [];
      const compatibleIndices = [];

      for ( let j = i + 1; j < this.renderableFaces.length; j++ ) {
        const otherRenderableFace = this.renderableFaces[ j ];

        if ( renderProgram.equals( otherRenderableFace.renderProgram ) ) {
          compatibleFaces.push( otherRenderableFace );
          compatibleIndices.push( j );
        }
      }

      if ( compatibleFaces.length ) {
        // TODO: something better than using LinearEdge.toPolygons, it is not high performance (we should trace edges,
        // TODO: like traceCombineFaces).
        const polygons = [ renderableFace, ...compatibleFaces ].map( renderableFace => renderableFace.face.toPolygonalFace().polygons ).flat();
        const simplifiedPolygons = LinearEdge.toPolygons( LinearEdge.fromPolygons( polygons ) );
        const newFace = new PolygonalFace( simplifiedPolygons );
        this.renderableFaces[ i ] = new RenderableFace(
          newFace,
          renderProgram,
          newFace.getBounds()
        );

        while ( compatibleIndices.length ) {
          this.renderableFaces.splice( compatibleIndices.pop()!, 1 );
        }
      }
    }
  }

  public updateWidthHeight( width: number, height: number ): void {
    assert && assert( Number.isInteger( width ) && Number.isInteger( height ) );

    this.width = width;
    this.height = height;

    this.bounds = new Bounds2( 0, 0, width, height );
    this.gridBounds = getPolygonFilterGridBounds( this.bounds, this.polygonFiltering );

    this.renderableFaces.length = 0;
    this.renderableFaces.push( new RenderableFace(
      PolygonalFace.fromBounds( this.bounds ),
      RenderColor.TRANSPARENT,
      this.bounds
    ) );

    // TODO: splitting of radial/linear gradients!!!
  }

  public getImageData( options?: CombinedRasterOptions ): ImageData {
    const raster = new CombinedRaster( this.width, this.height, combineOptions<CombinedRasterOptions>( {
      colorSpace: this.colorSpace
    }, options ) );

    // TODO: OMG, get rid of this! Get proper typing with logs so this isn't needed
    if ( assert ) {
      // @ts-expect-error
      window.debugData = {
        areas: []
      };
    }

    Rasterize.rasterizeAccumulate(
      raster,
      this.renderableFaces,
      this.bounds,
      this.gridBounds,
      Vector2.ZERO,
      this.polygonFiltering
    );

    return raster.toImageData();
  }

  public getCanvas( options?: CombinedRasterOptions ): HTMLCanvasElement {
    return Rasterize.imageDataToCanvas( this.getImageData( options ) );
  }
}


scenery.register( 'VectorCanvas', VectorCanvas );
