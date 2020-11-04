// Copyright 2020, University of Colorado Boulder

/**
 * A filter that can be represented by a single color matrix operation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import CanvasContextWrapper from './CanvasContextWrapper.js';
import Filter from './Filter.js';
import Utils from './Utils.js';

const isImageDataSupported = Utils.supportsImageDataCanvasFilter();

class ColorMatrixFilter extends Filter {
  /**
   * @param {number} m00
   * @param {number} m01
   * @param {number} m02
   * @param {number} m03
   * @param {number} m04
   * @param {number} m10
   * @param {number} m11
   * @param {number} m12
   * @param {number} m13
   * @param {number} m14
   * @param {number} m20
   * @param {number} m21
   * @param {number} m22
   * @param {number} m23
   * @param {number} m24
   * @param {number} m30
   * @param {number} m31
   * @param {number} m32
   * @param {number} m33
   * @param {number} m34
   */
  constructor( m00, m01, m02, m03, m04,
               m10, m11, m12, m13, m14,
               m20, m21, m22, m23, m24,
               m30, m31, m32, m33, m34  ) {

    assert && assert( typeof m00 === 'number' && isFinite( m00 ), 'm00 should be a finite number' );
    assert && assert( typeof m01 === 'number' && isFinite( m01 ), 'm01 should be a finite number' );
    assert && assert( typeof m02 === 'number' && isFinite( m02 ), 'm02 should be a finite number' );
    assert && assert( typeof m03 === 'number' && isFinite( m03 ), 'm03 should be a finite number' );
    assert && assert( typeof m04 === 'number' && isFinite( m04 ), 'm04 should be a finite number' );

    assert && assert( typeof m10 === 'number' && isFinite( m10 ), 'm10 should be a finite number' );
    assert && assert( typeof m11 === 'number' && isFinite( m11 ), 'm11 should be a finite number' );
    assert && assert( typeof m12 === 'number' && isFinite( m12 ), 'm12 should be a finite number' );
    assert && assert( typeof m13 === 'number' && isFinite( m13 ), 'm13 should be a finite number' );
    assert && assert( typeof m14 === 'number' && isFinite( m14 ), 'm14 should be a finite number' );

    assert && assert( typeof m20 === 'number' && isFinite( m20 ), 'm20 should be a finite number' );
    assert && assert( typeof m21 === 'number' && isFinite( m21 ), 'm21 should be a finite number' );
    assert && assert( typeof m22 === 'number' && isFinite( m22 ), 'm22 should be a finite number' );
    assert && assert( typeof m23 === 'number' && isFinite( m23 ), 'm23 should be a finite number' );
    assert && assert( typeof m24 === 'number' && isFinite( m24 ), 'm24 should be a finite number' );

    assert && assert( typeof m30 === 'number' && isFinite( m30 ), 'm30 should be a finite number' );
    assert && assert( typeof m31 === 'number' && isFinite( m31 ), 'm31 should be a finite number' );
    assert && assert( typeof m32 === 'number' && isFinite( m32 ), 'm32 should be a finite number' );
    assert && assert( typeof m33 === 'number' && isFinite( m33 ), 'm33 should be a finite number' );
    assert && assert( typeof m34 === 'number' && isFinite( m34 ), 'm34 should be a finite number' );

    super();

    // @public {number}
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
   * @public
   * @override
   *
   * @param {SVGFilterElement} svgFilter
   * @param {string} inName
   * @param {string} [resultName]
   */
  applySVGFilter( svgFilter, inName, resultName ) {
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
   * @public
   * @override
   *
   * @param {CanvasContextWrapper} wrapper
   */
  applyCanvasFilter( wrapper, resultWrapper ) {
    assert && assert( wrapper instanceof CanvasContextWrapper );

    const width = wrapper.canvas.width;
    const height = wrapper.canvas.height;

    const imageData = wrapper.context.getImageData( 0, 0, width, height );

    const size = width * height;
    for ( let i = 0; i < size; i++ ) {
      const index = i * 4;

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

    wrapper.context.putImageData( imageData, 0, 0 );
  }

  /**
   * @public
   * @override
   *
   * @returns {boolean}
   */
  isSVGCompatible() {
    return true;
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  isCanvasCompatible() {
    return super.isCanvasCompatible() || isImageDataSupported;
  }
}

scenery.register( 'ColorMatrixFilter', ColorMatrixFilter );
export default ColorMatrixFilter;