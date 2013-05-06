// Copyright 2002-2012, University of Colorado

/**
 * The main 'scenery' namespace object for the exported (non-Require.js) API. Used internally
 * since it prevents Require.js issues with circular dependencies.
 *
 * The returned scenery object namespace may be incomplete if not all modules are listed as
 * dependencies. Please use the 'main' module for that purpose if all of Scenery is desired.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  window.sceneryAssert = require( 'ASSERT/assert' )( 'scenery' );
  window.sceneryAssertExtra = require( 'ASSERT/assert' )( 'scenery.extra' );
  
  window.sceneryLayerLog = null; //function( ob ) { console.log( ob ); };
  
  // will be filled in by other modules
  return {};
} );
