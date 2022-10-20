// Copyright 2020-2022, University of Colorado Boulder

/**
 * A filter that can be represented by a single color matrix operation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import platform from '../../../phet-core/js/platform.js';
import { CanvasContextWrapper, Filter, scenery, Utils } from '../imports.js';

const isImageDataSupported = Utils.supportsImageDataCanvasFilter();
const useFakeGamma = platform.chromium;

export default class ColorMatrixFilter extends Filter {

  private m00: number;
  private m01: number;
  private m02: number;
  private m03: number;
  private m04: number;
  private m10: number;
  private m11: number;
  private m12: number;
  private m13: number;
  private m14: number;
  private m20: number;
  private m21: number;
  private m22: number;
  private m23: number;
  private m24: number;
  private m30: number;
  private m31: number;
  private m32: number;
  private m33: number;
  private m34: number;

  /**
   * NOTE: It is possible but not generally recommended to create custom ColorMatrixFilter types. They should be
   * compatible with Canvas and SVG, HOWEVER any WebGL/DOM content cannot work with those custom filters, and any
   * combination of multiple SVG or Canvas elements will ALSO not work (since there is no CSS filter function that can
   * do arbitrary color matrix operations). This means that performance will likely be reduced UNLESS all content is
   * within a single SVG block.
   *
   * Please prefer the named subtypes where possible.
   *
   * The resulting color is the result of the matrix multiplication:
   *
   * [ m00 m01 m02 m03 m04 ]   [ r ]
   * [ m10 m11 m12 m13 m14 ]   [ g ]
   * [ m20 m21 m22 m23 m24 ] * [ b ]
   * [ m30 m31 m32 m33 m34 ]   [ a ]
   *                           [ 1 ]
   */
  public constructor( m00: number, m01: number, m02: number, m03: number, m04: number,
               m10: number, m11: number, m12: number, m13: number, m14: number,
               m20: number, m21: number, m22: number, m23: number, m24: number,
               m30: number, m31: number, m32: number, m33: number, m34: number ) {

    assert && assert( isFinite( m00 ), 'm00 should be a finite number' );
    assert && assert( isFinite( m01 ), 'm01 should be a finite number' );
    assert && assert( isFinite( m02 ), 'm02 should be a finite number' );
    assert && assert( isFinite( m03 ), 'm03 should be a finite number' );
    assert && assert( isFinite( m04 ), 'm04 should be a finite number' );

    assert && assert( isFinite( m10 ), 'm10 should be a finite number' );
    assert && assert( isFinite( m11 ), 'm11 should be a finite number' );
    assert && assert( isFinite( m12 ), 'm12 should be a finite number' );
    assert && assert( isFinite( m13 ), 'm13 should be a finite number' );
    assert && assert( isFinite( m14 ), 'm14 should be a finite number' );

    assert && assert( isFinite( m20 ), 'm20 should be a finite number' );
    assert && assert( isFinite( m21 ), 'm21 should be a finite number' );
    assert && assert( isFinite( m22 ), 'm22 should be a finite number' );
    assert && assert( isFinite( m23 ), 'm23 should be a finite number' );
    assert && assert( isFinite( m24 ), 'm24 should be a finite number' );

    assert && assert( isFinite( m30 ), 'm30 should be a finite number' );
    assert && assert( isFinite( m31 ), 'm31 should be a finite number' );
    assert && assert( isFinite( m32 ), 'm32 should be a finite number' );
    assert && assert( isFinite( m33 ), 'm33 should be a finite number' );
    assert && assert( isFinite( m34 ), 'm34 should be a finite number' );

    super();

    this.m00 = m00;
    this.m01 = m01;
    this.m02 = m02;
    this.m03 = m03;
    this.m04 = m04;
    this.m10 = m10;
    this.m11 = m11;
    this.m12 = m12;
    this.m13 = m13;
    this.m14 = m14;
    this.m20 = m20;
    this.m21 = m21;
    this.m22 = m22;
    this.m23 = m23;
    this.m24 = m24;
    this.m30 = m30;
    this.m31 = m31;
    this.m32 = m32;
    this.m33 = m33;
    this.m34 = m34;
  }

  /**
   * Appends filter sub-elements into the SVG filter element provided. Should include an in=${inName} for all inputs,
   * and should either output using the resultName (or if not provided, the last element appended should be the output).
   * This effectively mutates the provided filter object, and will be successively called on all Filters to build an
   * SVG filter object.
   */
  public applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    Filter.applyColorMatrix(
      `${toSVGNumber( this.m00 )} ${toSVGNumber( this.m01 )} ${toSVGNumber( this.m02 )} ${toSVGNumber( this.m03 )} ${toSVGNumber( this.m04 )} ` +
      `${toSVGNumber( this.m10 )} ${toSVGNumber( this.m11 )} ${toSVGNumber( this.m12 )} ${toSVGNumber( this.m13 )} ${toSVGNumber( this.m14 )} ` +
      `${toSVGNumber( this.m20 )} ${toSVGNumber( this.m21 )} ${toSVGNumber( this.m22 )} ${toSVGNumber( this.m23 )} ${toSVGNumber( this.m24 )} ` +
      `${toSVGNumber( this.m30 )} ${toSVGNumber( this.m31 )} ${toSVGNumber( this.m32 )} ${toSVGNumber( this.m33 )} ${toSVGNumber( this.m34 )}`,
      svgFilter, inName, resultName
    );
  }

  /**
   * Given a specific canvas/context wrapper, this method should mutate its state so that the canvas now holds the
   * filtered content. Usually this would be by using getImageData/putImageData, however redrawing or other operations
   * are also possible.
   */
  public applyCanvasFilter( wrapper: CanvasContextWrapper ): void {
    const width = wrapper.canvas.width;
    const height = wrapper.canvas.height;

    const imageData = wrapper.context.getImageData( 0, 0, width, height );

    const size = width * height;
    for ( let i = 0; i < size; i++ ) {
      const index = i * 4;

      if ( useFakeGamma ) {
        // Gamma-corrected version, which seems to match SVG/DOM
        // Eek, this seems required for chromium Canvas to have a standard behavior?
        const gamma = 1.45;
        const r = Math.pow( imageData.data[ index + 0 ] / 255, gamma );
        const g = Math.pow( imageData.data[ index + 1 ] / 255, gamma );
        const b = Math.pow( imageData.data[ index + 2 ] / 255, gamma );
        const a = Math.pow( imageData.data[ index + 3 ] / 255, gamma );

        // Clamp/round should be done by the UInt8Array, we don't do it here for performance reasons.
        imageData.data[ index + 0 ] = 255 * Math.pow( r * this.m00 + g * this.m01 + b * this.m02 + a * this.m03 + this.m04, 1 / gamma );
        imageData.data[ index + 1 ] = 255 * Math.pow( r * this.m10 + g * this.m11 + b * this.m12 + a * this.m13 + this.m14, 1 / gamma );
        imageData.data[ index + 2 ] = 255 * Math.pow( r * this.m20 + g * this.m21 + b * this.m22 + a * this.m23 + this.m24, 1 / gamma );
        imageData.data[ index + 3 ] = 255 * Math.pow( r * this.m30 + g * this.m31 + b * this.m32 + a * this.m33 + this.m34, 1 / gamma );
      }
      else {
        const r = imageData.data[ index + 0 ];
        const g = imageData.data[ index + 1 ];
        const b = imageData.data[ index + 2 ];
        const a = imageData.data[ index + 3 ];

        // Clamp/round should be done by the UInt8Array, we don't do it here for performance reasons.
        imageData.data[ index + 0 ] = r * this.m00 + g * this.m01 + b * this.m02 + a * this.m03 + this.m04;
        imageData.data[ index + 1 ] = r * this.m10 + g * this.m11 + b * this.m12 + a * this.m13 + this.m14;
        imageData.data[ index + 2 ] = r * this.m20 + g * this.m21 + b * this.m22 + a * this.m23 + this.m24;
        imageData.data[ index + 3 ] = r * this.m30 + g * this.m31 + b * this.m32 + a * this.m33 + this.m34;
      }
    }

    wrapper.context.putImageData( imageData, 0, 0 );
  }

  public override isSVGCompatible(): boolean {
    return true;
  }

  public override isCanvasCompatible(): boolean {
    return super.isCanvasCompatible() || isImageDataSupported;
  }

  public getCSSFilterString(): string {
    throw new Error( 'unimplemented' );
  }
}

scenery.register( 'ColorMatrixFilter', ColorMatrixFilter );
