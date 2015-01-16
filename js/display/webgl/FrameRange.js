//  Copyright 2002-2014, University of Colorado Boulder

/**
 * Specifies the bounds region and the spriteSheetIndex,the frame belongs to.
 * @author Sharfudeen Ashraf
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );

  /**
   *
   * @param {Bounds2} bounds
   * @param {number} spriteSheetIndex
   * @constructor
   */
  function FrameRange( bounds, spriteSheetIndex ) {
    this.bounds = bounds;
    this.spriteSheetIndex = spriteSheetIndex;
  }

  return inherit( Object, FrameRange );

} );