// Copyright 2019-2020, University of Colorado Boulder

/**
 * Utilities specific to the keyboard for handling zoom/pan control.
 *
 * @author Jesse Greenberg
 */

import scenery from '../scenery.js';
import KeyboardUtils from './KeyboardUtils.js';

const KeyboardZoomUtils = {

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
    return KeyboardZoomUtils.isPlatformMac() ? 'metaKey' : 'ctrlKey';
  },

  /**
   * Returns true of the keyboard input indicates that a zoom command was initiated. Different keys are checked
   * on mac devices (which go through the Cmd key) and windows devices (which use the ctrl modifier).
   *
   * @param {Event} event
   * @param {boolean} zoomIn - do you want to check for zoom in or zoom out?
   * @returns {boolean}
   */
  isZoomCommand: ( event, zoomIn ) => {
    const zoomKey = zoomIn ? KeyboardUtils.KEY_EQUALS : KeyboardUtils.KEY_MINUS;
    const metaKey = KeyboardZoomUtils.getPlatformZoomMetaKey();
    return event[ metaKey ] && event.keyCode === zoomKey;
  },

  /**
   * Returns true if the keyboard command indicates a "zoom reset". This is ctrl + 0 on Win and cmd + 0 on mac.
   *
   * TODO: I suspect that these zoom specific functions should be moved out of KeyboardUtils.js
   * @param {Event} event
   * @returns {boolean}
   */
  isZoomResetCommand: event => {
    const metaKey = KeyboardZoomUtils.getPlatformZoomMetaKey();
    return event[ metaKey ] && event.keyCode === KeyboardUtils.KEY_0;
  }
};

scenery.register( 'KeyboardZoomUtils', KeyboardZoomUtils );
export default KeyboardZoomUtils;