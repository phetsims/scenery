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
    this.filterRegionPercentage = filterRegionPercentage;
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
   * @returns {SVGFilterElement}
   */
  createSVGFilter() {
    const svgFilter = super.createSVGFilter();

    // e.g. <feGaussianBlur stdDeviation="[radius radius]" edgeMode="[edge mode]" >
    const feGaussianBlur = document.createElementNS( svgns, 'feGaussianBlur' );
    feGaussianBlur.setAttribute( 'stdDeviation', toSVGNumber( this.standardDeviation ) );
    feGaussianBlur.setAttribute( 'edgeMode', 'none' ); // Don't pad things!
    svgFilter.appendChild( feGaussianBlur );

    // Bleh, no good way to handle the filter region? https://drafts.fxtf.org/filter-effects/#filter-region
    // If we WANT to track things by their actual display size AND pad pixels, AND copy tons of things... we could
    // potentially use the userSpaceOnUse and pad the proper number of pixels. That sounds like an absolute pain, AND
    // a performance drain and abstraction break.
    const min = `-${toSVGNumber( this.filterRegionPercentage )}%`;
    const size = `${toSVGNumber( 2 * this.filterRegionPercentage + 100 )}%`;
    svgFilter.setAttribute( 'x', min );
    svgFilter.setAttribute( 'y', min );
    svgFilter.setAttribute( 'width', size );
    svgFilter.setAttribute( 'height', size );

    return svgFilter;
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