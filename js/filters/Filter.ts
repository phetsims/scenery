// Copyright 2020-2022, University of Colorado Boulder

/**
 * Base type for filters
 *
 * Filters have different ways of being applied, depending on what the platform supports AND what content is below.
 * These different ways have potentially different performance characteristics, and potentially quality differences.
 *
 * The current ways are:
 * - DOM element with CSS filter specified (can include mixed content and WebGL underneath, and this is used as a
 *   general fallback). NOTE: General color matrix support is NOT provided under this, we only have specific named
 *   filters that can be used.
 * - SVG filter elements (which are very flexible, a combination of filters may be combined into SVG filter elements).
 *   This only works if ALL of the content under the filter(s) can be placed in one SVG element, so a layerSplit or
 *   non-SVG content can prevent this from being used.
 * - Canvas filter attribute (similar to DOM CSS). Similar to DOM CSS, but not as accelerated (requires applying the
 *   filter by drawing into another Canvas). Chromium-based browsers seem to have issues with the color space used,
 *   so this can't be used on that platform. Additionally, this only works if ALL the content under the filter(s) can
 *   be placed in one Canvas, so a layerSplit or non-SVG content can prevent this from being used.
 * - Canvas ImageData. This is a fallback where we directly get, manipulate, and set pixel data in a Canvas (with the
 *   corresponding performance hit that it takes to CPU-process every pixel). Additionally, this only works if ALL the
 *   content under the filter(s) can   be placed in one Canvas, so a layerSplit or non-SVG content can prevent this from
 *   being used.
 *
 * Some filters may have slightly different appearances depending on the browser/platform/renderer.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { CanvasContextWrapper, Features, scenery, svgns } from '../imports.js';

let globalId = 1;

export default abstract class Filter {

  // (scenery-internal)
  public readonly id: string;

  // Can be mutated by subtypes, determines what filter region increases should be used for when SVG is used for
  // rendering.
  public filterRegionPercentageIncrease: number;

  public constructor() {
    this.id = `filter${globalId++}`;
    this.filterRegionPercentageIncrease = 0;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   */
  public abstract getCSSFilterString(): string;

  /**
   * Appends filter sub-elements into the SVG filter element provided. Should include an in=${inName} for all inputs,
   * and should either output using the resultName (or if not provided, the last element appended should be the output).
   * This effectively mutates the provided filter object, and will be successively called on all Filters to build an
   * SVG filter object.
   */
  public abstract applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void;

  /**
   * Given a specific canvas/context wrapper, this method should mutate its state so that the canvas now holds the
   * filtered content. Usually this would be by using getImageData/putImageData, however redrawing or other operations
   * are also possible.
   */
  public abstract applyCanvasFilter( wrapper: CanvasContextWrapper ): void;

  public isDOMCompatible(): boolean {
    // TODO: We can browser-check on things like color matrix? But we want to disallow things that we can't guarantee we
    // can support?
    return false;
  }

  public isSVGCompatible(): boolean {
    return false;
  }

  public isCanvasCompatible(): boolean {
    return Features.canvasFilter ? this.isDOMCompatible() : false;
  }

  public isWebGLCompatible(): boolean {
    return false;
  }

  /**
   * Returns a string form of this object
   */
  public toString(): string {
    return this.id;
  }

  /**
   * Applies a color matrix effect into an existing SVG filter.
   */
  public static applyColorMatrix( matrixValues: string, svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    const feColorMatrix = document.createElementNS( svgns, 'feColorMatrix' );

    feColorMatrix.setAttribute( 'type', 'matrix' );
    feColorMatrix.setAttribute( 'values', matrixValues );
    feColorMatrix.setAttribute( 'in', inName );

    // Since the DOM effects are done with sRGB and we can't manipulate that, we'll instead adjust SVG to apply the
    // effects in sRGB so that we have consistency
    feColorMatrix.setAttribute( 'color-interpolation-filters', 'sRGB' );

    if ( resultName ) {
      feColorMatrix.setAttribute( 'result', resultName );
    }
    svgFilter.appendChild( feColorMatrix );
  }
}

scenery.register( 'Filter', Filter );
