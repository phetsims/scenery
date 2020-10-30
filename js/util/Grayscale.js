// Copyright 2020, University of Colorado Boulder

/**
 * Grayscale filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import Filter from './Filter.js';

class Grayscale extends Filter {
  /**
   * @param {number} [amount]
   */
  constructor( amount = 1 ) {
    assert && assert( typeof amount === 'number', 'Grayscale amount should be a number' );
    assert && assert( isFinite( amount ), 'Grayscale amount should be finite' );
    assert && assert( amount >= 0, 'Grayscale amount should be non-negative' );
    assert && assert( amount <= 1, 'Grayscale amount should be no greater than 1' );

    super();

    // @public {number}
    this.amount = amount;
  }

  /**
   * @public
   * @override
   *
   * @returns {string}
   */
  getCSSFilterString() {
    return `grayscale(${toSVGNumber( this.amount )})`;
  }

  /**
   * @public
   * @override
   *
   * @param {SVGFilterElement} svgFilter
   * @param {string} inName
   * @param {string} [resultName]
   */
  applySVGFilter( svgFilter, inName, resultName ) {

    /*
     * According to the spec:
     * <filter id="grayscale">
     *   <feColorMatrix type="matrix"
     *              values="
     *     (0.2126 + 0.7874 * [1 - amount]) (0.7152 - 0.7152  * [1 - amount]) (0.0722 - 0.0722 * [1 - amount]) 0 0
     *     (0.2126 - 0.2126 * [1 - amount]) (0.7152 + 0.2848  * [1 - amount]) (0.0722 - 0.0722 * [1 - amount]) 0 0
     *     (0.2126 - 0.2126 * [1 - amount]) (0.7152 - 0.7152  * [1 - amount]) (0.0722 + 0.9278 * [1 - amount]) 0 0
     *     0 0 0 1 0"/>
     * </filter>
     */

    const n = 1 - this.amount;

    Filter.applyColorMatrix(
      `${toSVGNumber( 0.2126 + 0.7874 * n )} ${toSVGNumber( 0.7152 - 0.7152  * n )} ${toSVGNumber( 0.0722 - 0.0722 * n )} 0 0 ` +
      `${toSVGNumber( 0.2126 - 0.2126 * n )} ${toSVGNumber( 0.7152 + 0.2848  * n )} ${toSVGNumber( 0.0722 - 0.0722 * n )} 0 0 ` +
      `${toSVGNumber( 0.2126 - 0.2126 * n )} ${toSVGNumber( 0.7152 - 0.7152  * n )} ${toSVGNumber( 0.0722 + 0.9278 * n )} 0 0 ` +
      '0 0 0 1 0',
      svgFilter, inName, resultName
    );
  }

  /**
   * @public
   * @override
   *
   * @returns {*}
   */
  isDOMCompatible() {
    return true;
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
}

scenery.register( 'Grayscale', Grayscale );
export default Grayscale;