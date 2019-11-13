// Copyright 2019, University of Colorado Boulder

/**
 * Utilities specific to the keyboard for handling zoom/pan control.
 *
 * @author Jesse Greenberg
 */
define( require => {
  'use strict';

  // modules
  const KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  const scenery = require( 'SCENERY/scenery' );

  const KeyboardZoomUtil = {

    /**
     * Returns true if the platform is most likely a Mac device. Pan/Zoom will use different modifier keys in this case.
     *
     * TODO: Move to platform if generally useful?
     *
     * @returns {boolean}
     */
    isPlatformMac: () => {
      return _.includes( window.navigator.platform, 'Mac' );
    },

    /**
     * Get the 'meta' key for the platform that would indicate user wants to zoom. This is 'metaKey' on Mac and 'ctrl'
     * on Windows.
     *
     * @returns {string}
     */
    getPlatformZoomMetaKey: () => {
      return KeyboardZoomUtil.isPlatformMac() ? 'metaKey' : 'ctrlKey';
    },

    /**
     * Returns true of the keyboard input indicates that a zoom command was initiated. Different keys are checked
     * on mac devices (which go through the Cmd key) and windows devices (which use the ctrl modifier).
     *
     * @param {DOMEvent} event
     * @param {boolean} zoomIn - do you want to check for zoom in or zoom out?
     * @returns {boolean}
     */
    isZoomCommand: ( event, zoomIn ) => {
      const zoomKey = zoomIn ? KeyboardUtil.KEY_EQUALS : KeyboardUtil.KEY_MINUS;
      const metaKey = KeyboardZoomUtil.getPlatformZoomMetaKey();
      return event[ metaKey ] && event.keyCode === zoomKey;
    },

    /**
     * Returns true if the keyboard command indicates a "zoom reset". This is ctrl + 0 on Win and cmd + 0 on mac.
     *
     * TODO: I suspect that these zoom specific functions should be moved out of KeyboardUtil.js
     * @param {DOMEvent} event
     * @returns {boolean}
     */
    isZoomResetCommand: event => {
      const metaKey = KeyboardZoomUtil.getPlatformZoomMetaKey();
      return event[ metaKey ] && event.keyCode === KeyboardUtil.KEY_0;
    }
  };

  return scenery.register( 'KeyboardZoomUtil', KeyboardZoomUtil );
} );
