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
    }

    /**
     * Update any AccessiblePeers that are in the list of dirtyDescendants. Updating AccessiblePeers involves
     * DOM operations so we sometimes hide the root element of the PDOM first to prevent the browser from
     * doing expensive layout calculations for every single update. This way the browser should do work for
     * layout/reflow at most twice per animation frame.
     *
     * TODO: Optimize to hide only dirty portions of the PDOM? https://github.com/phetsims/scenery/issues/663
     */
    updateDirtyDescendantContent() {

      // if multiple elements are marked as dirty we hide the PDOM to prevent layout calculations (thrashing) as
      // the document is modified - the hide will also cause a layout recalculation, so we only do this
      // if the browser will have more than a number of updates
      var preventThrashing = this.dirtyDescendants.length > 1;
      if ( preventThrashing ) {
        // console.log( 'prevent thrash, hiding PDOM')
        this.display.accessibleDOMElement.style.visibility = 'hidden';
      }

      while ( this.dirtyDescendants.length ) {
        this.dirtyDescendants.pop().updateDirty();
      }

      if ( preventThrashing ) {
        // console.log( 'prevent thrash, showing PDOM');
        this.display.accessibleDOMElement.style.visibility = 'visible';
      }
    }
  }

  return scenery.register( 'RootAccessiblePeer', RootAccessiblePeer );
} );
