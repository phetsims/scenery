// Copyright 2018-2019, University of Colorado Boulder

/**
 * "definition" type for generalized color paints (anything that can be given to a fill/stroke that represents just a
 * solid color). Does NOT include any type of gradient or pattern.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const Color = require( 'SCENERY/util/Color' );
  const Property = require( 'AXON/Property' );
  const scenery = require( 'SCENERY/scenery' );

  const ColorDef = {
    /**
     * Returns whether the parameter is considered to be a ColorDef.
     * @public
     *
     * @param {*} color
     * @returns {boolean}
     */
    isColorDef: function( color ) {
      return color === null ||
             typeof color === 'string' ||
             color instanceof Color ||
             ( color instanceof Property && (
               color.value === null ||
               typeof color.value === 'string' ||
               color.value instanceof Color
             ) );
    }
  };

  scenery.register( 'ColorDef', ColorDef );

  return ColorDef;
} );
