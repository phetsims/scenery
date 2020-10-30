// Copyright 2020, University of Colorado Boulder

/**
 * GaussianBlur filter
 *
 * TODO: preventFit OR handle bounds increase (or both)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import Filter from './Filter.js';
import svgns from './svgns.js';

class GaussianBlur extends Filter {
  /**
   * @param {number} standardDeviation
   * @param {number} [filterRegionPercentage]
   */
  constructor( standardDeviation, filterRegionPercentage = 15 ) {
    assert && assert( typeof standardDeviation === 'number', 'GaussianBlur standardDeviation should be a number' );
    assert && assert( isFinite( standardDeviation ), 'GaussianBlur standardDeviation should be finite' );
    assert && assert( standardDeviation >= 0, 'GaussianBlur standardDeviation should be non-negative' );

    super();

    // @public {number}
    this.standardDeviation = standardDeviation;

    this.filterRegionPercentageIncrease = filterRegionPercentage;
  }

  /**
   * @public
   * @override
   *
   * @returns {string}
   */
  getCSSFilterString() {
    return `blur(${toSVGNumber( this.standardDeviation )}px)`;
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
    // e.g. <feGaussianBlur stdDeviation="[radius radius]" edgeMode="[edge mode]" >
    const feGaussianBlur = document.createElementNS( svgns, 'feGaussianBlur' );
    feGaussianBlur.setAttribute( 'stdDeviation', toSVGNumber( this.standardDeviation ) );
    feGaussianBlur.setAttribute( 'edgeMode', 'none' ); // Don't pad things!
    svgFilter.appendChild( feGaussianBlur );

    feGaussianBlur.setAttribute( 'in', inName );
    if ( resultName ) {
      feGaussianBlur.setAttribute( 'result', resultName );
    }
    svgFilter.appendChild( feGaussianBlur );
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

scenery.register( 'GaussianBlur', GaussianBlur );
export default GaussianBlur;