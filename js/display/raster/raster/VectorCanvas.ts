// Copyright 2023, University of Colorado Boulder

/**
 * Similar to an DOM Canvas, but stores a vector representation of the relevant drawing commands.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { CombinedRaster, CombinedRasterOptions, getPolygonFilterGridBounds, PolygonalBoolean, PolygonalFace, PolygonClipping, PolygonFilterType, Rasterize, RenderableFace, RenderColor, RenderColorSpace, RenderPath, RenderStack, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { combineOptions } from '../../../../../phet-core/js/optionize.js';

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
    // TODO: COLOR SPACE!!!! (We need to know, so we can

    this.updateWidthHeight( width, height );
  }

  // Assumes sRGB? We'll do srgb-linear blending to mimic for now?
  public fillColor( renderPath: RenderPath, color: Vector4 ): void {

    const bounds = renderPath.getBounds();

    renderPath = new RenderPath( renderPath.fillRule, renderPath.subpaths.map( subpath => PolygonClipping.boundsClipPolygon( subpath, this.bounds ) ) );

    const newRenderableFaces: RenderableFace[] = [];

    const renderProgram = new RenderColor( color ).colorConverted( RenderColorSpace.sRGB, this.colorSpace === 'srgb' ? RenderColorSpace.premultipliedSRGB : RenderColorSpace.premultipliedDisplayP3 );

    for ( let i = 0; i < this.renderableFaces.length; i++ ) {
      const renderableFace = this.renderableFaces[ i ];

      if ( renderableFace.bounds.intersectsBounds( bounds ) ) {
        const existingRenderPath = new RenderPath( 'nonzero', renderableFace.face.toPolygonalFace().polygons );
        const overlaps = PolygonalBoolean.getOverlaps( existingRenderPath, renderPath );

        // TODO: Should we check faces for area?
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

    // TODO: something better than using LinearEdge.toPolygons, it is not high performance (we should trace edges,
    // TODO: like traceCombineFaces).
    // TODO: Use LinearEdge.toPolygons( LinearEdge.fromPolygons( polygons ) );
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
