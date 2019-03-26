// Copyright 2018, University of Colorado Boulder

/**
 * The AccessiblePeer associated with the root AccessibleInstance of the AccessibleInstance tree.
 * 
 * @author Jesse Greenberg
 */
define( ( require ) => {
  'use strict';

  // modules
  const AccessibleSiblingStyle = require( 'SCENERY/accessibility/AccessibleSiblingStyle' );
  const AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );
  const cleanArray = require( 'PHET_CORE/cleanArray' );
  const scenery = require( 'SCENERY/scenery' );

  // constants
  // number of frames before updating dirty AccessiblePeers - small value will update PDOM more frequently but will
  // result in more frequent DOM operations and (potentially) slower graphical animations
  const FRAMES_PER_UPDATE = 20;

  class RootAccessiblePeer extends AccessiblePeer {

    /**
     * Just calls initializeAccessiblePeer because AccessiblePeer mixes Poolable.
     *
     * @param {AccessibleInstance} accessibleInstance - root AccessibleInstance
     * @param {HTMLElement} rootElement - root of the PDOM
     */
    constructor( accessibleInstance, rootElement ) {
      assert && assert( accessibleInstance.isRootInstance, 'root AccessiblePeer should be associated with root AccessibleInstance' );

      // directly to initializeAccessiblePeer because AccessiblePeer mixes Poolable
      super( accessibleInstance, rootElement );

      // @private - root HTML element of the PDOM
      this._primarySibling = rootElement;

      // makes the HTML tree invisible while still interactive with AT
      this._primarySibling.className = AccessibleSiblingStyle.ROOT_CLASS_NAME;

      // @private {Array.<AccessiblePeer>} - list of all AccessiblePeers that need to be updated next updateDisplay
      this.dirtyDescendants = cleanArray( this.dirtyDescendants );

      // @private {boolean} - whether or not the dirty AccessiblePeers will be updated every updateDisply or less
      // frequently than that (for improved performance)
      this.leanAccessibilityChanges = this.display._leanAccessibilityChanges;
    }

    /**
     * Update any AccessiblePeers that are in the list of dirtyDescendants.
     * @public
     *
     * @param {number} frameID - identifier set in Display.updateDisplay, increments each update
     */
    updateDirtyDescendantContent( frameID ) {

      if ( this.leanAccessibilityChanges ) {

        // update the PDOM ~3 times per second assuming 60 fps
        if ( frameID % FRAMES_PER_UPDATE === 0 ) {
          this.cleanDirtyList();
        }
      }
      else {
        this.cleanDirtyList();
      }
    }

    /**
     * Update all AccessiblePeers in the dirty list (synchronously). 
     *
     * @private
     */
    cleanDirtyList() {
      while ( this.dirtyDescendants.length ) {
        this.dirtyDescendants.pop().updateDirty();
      }
    }
  }

  return scenery.register( 'RootAccessiblePeer', RootAccessiblePeer );
} );
